import os
import sys
import numpy as np
from glob import glob

from Rover.utils.logger import safe_print
from Rover.algos.ppo.ppo import PPO_Rover
from Rover.utils.env_register import register_rover_environments
from Rover.utils.env_util import make_vec_env
from Rover.utils.arg_parser import common_arg_parser, parse_unknown_args, play_only_arg_parser

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
    play_arg_parser = play_only_arg_parser()
    play_args, unknown_play_args = play_arg_parser.parse_known_args(args)
    extra_play_args = parse_cmdline_kwargs(unknown_play_args)


    if play_args.play_path is None:
        return
    play_path = glob(os.path.expanduser(play_args.play_path))
    assert len(play_path) == 1, "more than one folder matches with the patern given in --play_path"
    play_path = play_path[0]
    if not os.path.exists(play_path):
        return

    monitor_kwargs = dict(info_keywords=['death', 'goal_reached_flag', 'timeout'], reset_keywords=['current_goal'])

    if isinstance(play_args.play_goals, str):
        play_args.play_goals = eval(play_args.play_goals)

    goals = play_args.play_goals
    num_play_eps = play_args.num_play_eps
    nets = []
    if play_path.endswith('.zip'):  # is a single network
        nets.append(play_path)
        play_exp_dir = play_path.split('/saved_networks')[0]

    else:  # might be an exp folder
        maybe_exp_dir = os.listdir(play_path)
        if 'progress.csv' in maybe_exp_dir:  # is an exp folder
            play_exp_dir = play_path
            play_dir = os.path.join(play_exp_dir, 'saved_networks')
            if os.path.exists(play_dir):
                for content in os.listdir(play_dir):
                    if content.endswith('.zip'):
                        nets.append(os.path.join(play_dir, content))
            else:  # no network to play
                return
        else:  # is not exp folder or single network
            return

    with open(os.path.join(play_exp_dir, 'exp_call_args.txt')) as exp_call_file:
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
    env_kwargs = {"render_mode": "human", 'gamma': exp_args.gamma,
                      'start_at_initpos': exp_args.start_at_initpos,
                      'end_after_current_goal': not exp_args.dont_end_after_current_goal,
                      'random_current_goal': not exp_args.dont_random_current_goal}

    num_environments = exp_args.num_env
    env_kwargs.update(extra_exp_args)
    if rover_env.startswith('Rover4We'):
        env_kwargs.update(
            {"width": 440, "height": 270, "save_images": False, "verbose": exp_args.env_verbose,
             "img_reduced_size": exp_args.img_red_size})
        if 'Encoded' in rover_env:  #
            assert exp_args.encoder_name is not None, 'This environment MUST receive a proper encoder name to work'
            env_kwargs.update({'encoder_name': exp_args.encoder_name})

    # RECREATE THE EXP'S ENV
    envs = make_vec_env(rover_env, env_kwargs=env_kwargs, n_envs=num_environments, monitor_kwargs=monitor_kwargs)

    # PLAY
    error_loading = []
    for net in nets:
        net_num = net.split('/')[-1].split('.')[0]
        try:
            model = PPO_Rover.load(path=net)
        except:
            error_loading.append(net)
            safe_print('Network {} skipped due to loading error.'.format(net_num))
            continue
        for goal in goals:
            if goal == -1:
                envs.env_method('set_init_configs_full')
                safe_print("Playing Network {} in complete mode".format(net_num))
            else:
                safe_print("Playing Network {} in goal_{} mode".format(net_num, goal))
            if goal == 0:
                envs.env_method('set_init_configs_g0')
            if goal == 1:
                envs.env_method('set_init_configs_g1')
            if goal == 2:
                envs.env_method('set_init_configs_g2')
            observations = envs.reset()
            episode_counts = np.zeros(num_environments, dtype="int")

            if num_play_eps == -1:
                num_play_eps = np.inf
            episode_count_targets = np.array([(num_play_eps + i) // num_environments for i in range(num_environments)],
                                             dtype="int")
            while (episode_counts < episode_count_targets).any():
                actions, _ = model.predict(observation=observations, deterministic=True)
                observations, rewards, dones, infos = envs.step(actions)
                envs.render()
                for i in range(num_environments):
                    if episode_counts[i] < episode_count_targets[i]:
                        if dones[i]:
                            episode_counts[i] += 1

    if len(error_loading) > 0:
        safe_print('Network(s) {} skipped due to loading error.'.format(', '.join(error_loading)))
    envs.close()
    print()


if __name__ == "__main__":
    main(sys.argv)

