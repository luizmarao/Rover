import os
import sys
import numpy as np
from tqdm import tqdm
from glob import glob
import csv

from Rover.utils.logger import safe_print
from Rover.utils.evaluation import evaluate_policy
from Rover.algos.ppo.ppo import PPO_Rover
from Rover.utils.env_register import register_rover_environments
from Rover.utils.env_util import make_vec_env
from Rover.utils.arg_parser import common_arg_parser, parse_unknown_args, eval_only_arg_parser

register_rover_environments()


def parse_cmdline_kwargs(args):
    '''
    convert a list of '='-spaced command-line arguments to a dictionary, evaluating python objects when possible
    '''
    def parse(v):

        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v

    return {k: parse(v) for k, v in parse_unknown_args(args).items()}


def main(args):
    args_ = args
    exp_arg_parser = common_arg_parser()
    eval_arg_parser = eval_only_arg_parser()
    eval_args, unknown_eval_args = eval_arg_parser.parse_known_args(args)
    extra_eval_args = parse_cmdline_kwargs(unknown_eval_args)


    if eval_args.eval_exp_dir is None:
        return
    eval_exp_dir = glob(os.path.expanduser(eval_args.eval_exp_dir))
    assert len(eval_exp_dir) == 1, "more than one folder matches with the patern given in --eval_exp_dir"
    eval_exp_dir = eval_exp_dir[0]
    if not os.path.exists(eval_exp_dir):
        return



    if not eval_args.play:
        os.environ["MUJOCO_GL"] = 'egl'  # Set mujoco rendering to dedicated GPU


    monitor_kwargs = dict(info_keywords=['death', 'goal_reached_flag', 'timeout'], reset_keywords=['current_goal'])

    if isinstance(eval_args.eval_goals, str):
        eval_args.eval_goals = eval(eval_args.eval_goals)

    goals = eval_args.eval_goals
    num_eval_eps = eval_args.num_eval_eps
    maybe_exp_list = os.listdir(eval_exp_dir)
    eval_dirs = []
    exps_dirs = []
    num_nets = 0
    if 'progress.csv' in maybe_exp_list:  # is just one exp folder
        exp_dir = eval_exp_dir
        eval_dir = os.path.join(eval_exp_dir, 'saved_networks')
        if os.path.exists(eval_dir):
            eval_dirs.append(eval_dir)
            exps_dirs.append(exp_dir)
            nets = [1 if net.endswith('.zip') else 0 for net in os.listdir(eval_dir)]
            num_nets += sum(nets)
    else:  # might be a root folder for multiple exps
        for content in maybe_exp_list:
            content_path = os.path.join(eval_exp_dir, content)
            if os.path.isdir(content_path):
                if 'progress.csv' in os.listdir(content_path):  # Is an Exp Folder, search for saved data
                    exp_dir = content_path
                    eval_dir = os.path.join(content_path, 'saved_networks')
                    if os.path.exists(eval_dir):
                        eval_dirs.append(eval_dir)
                        exps_dirs.append(exp_dir)
                        nets = [1 if net.endswith('.zip') else 0 for net in os.listdir(eval_dir)]
                        num_nets += sum(nets)
    pbar_full = tqdm(desc='Total Evaluation Episodes', total=num_eval_eps * num_nets * len(goals), position=2)
    for idx, exp_dir in enumerate(exps_dirs):
        if len(exps_dirs) > 1:
            exp_name = exp_dir.split('/')[-1].split('-')[0]
            safe_print((' '*5+"##### BEGINING EXP {} EVALUATION #####"+' '*5).format(exp_name))
        with open(os.path.join(exp_dir, 'exp_call_args.txt')) as exp_call_file:
            # second line contains the call line, but start with 'python -m Rover.run' and end with '\n'
            exp_call_line = exp_call_file.readlines()[1][20:-2]
            exp_args = exp_call_line.split('--')[1:]  # separate args
            exp_args = ['--' + arg.replace(' ', '') for arg in exp_args]
        # ENVIRONMENT PARAMETERS
        exp_args, unknown_exp_args = exp_arg_parser.parse_known_args(exp_args)
        extra_exp_args = parse_cmdline_kwargs(unknown_exp_args)
        if isinstance(exp_args.img_red_size, str):
            exp_args.img_red_size = eval(exp_args.img_red_size)
        rover_env = exp_args.env
        env_kwargs = {"render_mode": "rgb_array", 'gamma': exp_args.gamma,
                          'start_at_initpos': exp_args.start_at_initpos,
                          'end_after_current_goal': not exp_args.dont_end_after_current_goal,
                          'random_current_goal': not exp_args.dont_random_current_goal}
        if eval_args.play:
            env_kwargs['render_mode'] = 'human'
        num_environments = exp_args.num_env
        env_kwargs.update(extra_exp_args)
        if rover_env.startswith('Rover4We'):
            env_kwargs.update(
                {"width": 440, "height": 270, "save_images": exp_args.save_images, "verbose": exp_args.env_verbose,
                 "img_reduced_size": exp_args.img_red_size})
            if 'Encoded' in rover_env:  #
                assert exp_args.encoder_name is not None, 'This environment MUST receive a proper encoder name to work'
                env_kwargs.update({'encoder_name': exp_args.encoder_name})

        # RECREATE THE EXP'S ENV
        envs = make_vec_env(rover_env, env_kwargs=env_kwargs, n_envs=num_environments, monitor_kwargs=monitor_kwargs)

        # RUN EVAL PHASE FOR THE EXP
        error_loading = []
        eval_results = {}
        for goal in goals:
            eval_results[goal] = []
        eval_nets_list = os.listdir(eval_dirs[idx])
        for net in eval_nets_list:
            if not net.endswith('.zip'):
                continue
            net_path = os.path.join(eval_dirs[idx], net)
            net_num = net.split('.')[0]
            try:
                model = PPO_Rover.load(path=net_path)
            except:
                error_loading.append(net)
                safe_print('Network {} skipped due to loading error.'.format(net))
                continue
            results = evaluate_policy(model=model, env=envs, net_num=net_num, goals=goals, n_eval_episodes=num_eval_eps,
                                      progress_bar_full=pbar_full, render=eval_args.play, deterministic=deterministic)
            for goal in goals:
                eval_results[goal].append(results[goal])

        # TREATS ACQUIRED DATA BEFORE WRITING EXP'S RESULTS
        for goal in goals:
            if goal == -1:
                file_path = os.path.join(exp_dir, 'Eval_complete_mode.csv')
            else:
                file_path = os.path.join(exp_dir, 'Eval_goal{}_mode.csv'.format(goal))
            file = open(file_path, 'w')
            for net_data in eval_results[goal]:
                gr_eps = net_data.pop('goal_reached_episodes')
                if goal == -1:
                    for g in range(3):
                        net_data['g{}_reached_eps'.format(g)] = gr_eps[g]
                        net_data['g{}_reached_eps (%)'.format(g)] = 100 * gr_eps[g] / net_data['total_eps']
                else:
                    net_data['g{}_reached_eps'.format(goal)] = gr_eps[goal]
                    net_data['g{}_reached_eps (%)'.format(goal)] = 100 * gr_eps[goal] / net_data['total_eps']

            writer = csv.DictWriter(file, eval_results[goal][0].keys())
            writer.writeheader()
            writer.writerows(eval_results[goal])
            file.close()
        if len(error_loading) > 0:
            safe_print('Network(s) {} skipped due to loading error.'.format(', '.join(error_loading)))
        envs.close()  # CLOSE ENV FOR NEW RUN IF MORE EXPS TO RUN
        print()


if __name__ == "__main__":
    main(sys.argv)

