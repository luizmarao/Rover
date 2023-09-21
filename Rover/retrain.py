import os
import sys
import numpy as np
import stable_baselines3.common.logger
from tqdm import tqdm
from glob import glob
import csv

from Rover.utils.logger import safe_print, configure
from Rover.utils.evaluation import evaluate_policy
from Rover.algos.ppo.ppo import PPO_Rover
from Rover.utils.env_register import register_rover_environments
from Rover.utils.env_util import make_vec_env
from Rover.utils.arg_parser import common_arg_parser, parse_unknown_args, retrain_arg_parser
from Rover.utils.lr_schedulers import linear_schedule
from Rover.utils.networks_ranking import RoverRankingSystem

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
    retrain_args_parser = retrain_arg_parser()
    retrain_args, unknown_retrain_args = retrain_args_parser.parse_known_args(args)
    extra_retrain_args = parse_cmdline_kwargs(unknown_retrain_args)

    # eval_arg_parser = eval_only_arg_parser()
    # eval_args, unknown_eval_args = eval_arg_parser.parse_known_args(args)
    # extra_eval_args = parse_cmdline_kwargs(unknown_eval_args)
    if retrain_args.exp_root_folder is None:
        return

    maybe_exp_folder = glob(os.path.expanduser(retrain_args.exp_root_folder))
    assert len(maybe_exp_folder) == 1, "more than one folder matches with the patern given in --exp_root_folder"
    maybe_exp_folder = maybe_exp_folder[0]
    if not os.path.exists(maybe_exp_folder):
        return

    os.environ["MUJOCO_GL"] = 'egl'  # Set mujoco rendering to dedicated GPU
    monitor_kwargs = dict(info_keywords=['death', 'goal_reached_flag', 'timeout'], reset_keywords=['current_goal'])


    maybe_exp_folder_content = os.listdir(maybe_exp_folder)
    if 'progress.csv' in maybe_exp_folder_content:  # is just one exp folder
        exp_dir = maybe_exp_folder
        nets_dir = os.path.join(exp_dir, 'saved_networks')
        if not os.path.exists(nets_dir):
            return
    else:
        return

    with open(os.path.join(exp_dir, 'exp_call_args.txt')) as exp_call_file:
        # second line contains the call line, but start with 'python -m Rover.run' and end with '\n'
        exp_call_line = exp_call_file.readlines()[1][20:-2]
        exp_args = exp_call_line.split('--')[1:]  # separate args
        exp_args = ['--' + arg.replace(' ', '') for arg in exp_args]
        # ENVIRONMENT PARAMETERS
        exp_arg_parser = common_arg_parser()
        exp_args, unknown_exp_args = exp_arg_parser.parse_known_args(exp_args)
        extra_exp_args = parse_cmdline_kwargs(unknown_exp_args)

        try:
            if isinstance(exp_args.img_red_size, str):
                exp_args.img_red_size = eval(exp_args.img_red_size)
        except:
            pass

        ## ENVIRONMENT PARAMETERS ##
        rover_env = exp_args.env
        env_kwargs = {"render_mode": "rgb_array", 'gamma': exp_args.gamma,
                          'start_at_initpos': exp_args.start_at_initpos,
                          'end_after_current_goal': not exp_args.dont_end_after_current_goal,
                          'random_current_goal': not exp_args.dont_random_current_goal}
        env_kwargs.update(extra_exp_args)
        num_environments = exp_args.num_env

        if rover_env.startswith('Rover4We'):
            env_kwargs.update(
                {"width": 440, "height": 270, "save_images": exp_args.save_images, "verbose": exp_args.env_verbose,
                 "img_reduced_size": exp_args.img_red_size})
            if 'Encoded' in rover_env:  #
                assert exp_args.encoder_name is not None, 'This environment MUST receive a proper encoder name to work'
                try:
                    exp_args.encoder_name = exp_args.encoder_name[0:2]+' '+exp_args.encoder_name[2:]  # Fix space removal from args loading
                except:
                    print('Problem loading given encoder name')
                env_kwargs.update({'encoder_name': exp_args.encoder_name})

        # RECREATE THE EXP'S ENV
        envs = make_vec_env(rover_env, env_kwargs=env_kwargs, n_envs=num_environments, monitor_kwargs=monitor_kwargs)

        ## MODEL LOAD AND RETRAIN ##
        total_retrain_timesteps = int(retrain_args.num_timesteps)

        nets_list = os.listdir(nets_dir)
        nets = []
        for net in nets_list:
            if not net.endswith('.zip'):
                continue
            nets.append(net.split('.')[0])

        nets.sort()

        net_path = os.path.join(nets_dir, nets[-1])

        try:
            model = PPO_Rover.load(path=net_path)
        except:
            safe_print('Loading error')
            return


        # MUST COPY CSV DATA BEFORE CONFIGURING THE LOGGER
        logger = configure(folder=None, format_strings=["stdout", "csv"])

        csv_writer = logger.output_formats[1]
        assert isinstance(csv_writer, stable_baselines3.common.logger.CSVOutputFormat)
        csv_writer.file.close()

        # LOADS THE CURRENT PROGRESS FILE UNTIL THE LATEST NETWORK'S LINE
        progress_file = open(os.path.join(exp_dir, 'progress.csv'), 'r')
        lines = progress_file.readlines()
        progress_file.close()
        csv_writer.file = open(os.path.join(exp_dir, 'progress.csv'), 'w+t')
        csv_writer.file.writelines(lines[:int(nets[-1]) + 1])
        csv_writer.file.flush()

        model.set_logger(logger)

        # MUST FIX LEARNING RATE SCHEDULE

        # MUST SET AND LOAD VALUES FOR ROVER RANKIG SYSTEM
        rover_rankings = RoverRankingSystem(networks_limit_per_ranking=exp_args.networks_limit_per_ranking,
                                            num_rankings=exp_args.num_rankings, save_path=logger.get_dir(),
                                            networks_subfolder='saved_networks', verbose=0)


        ranking_nets = {str(int(net_num)): rover_rankings.Network(int(net_num), [-np.inf]*3) for net_num in nets}
        for ranking_file_n in [1, 2, 3]:
            with open(os.path.join(exp_dir, 'networks_ranking{}.txt'.format(ranking_file_n))) as ranking_file:
                lines = ranking_file.readlines()
                for line in lines[1:]:
                    performance, iteration = line[:-1].split('\t')
                    ranking_nets[iteration].performance_list[ranking_file_n] = float(performance)

        model.set_ranking_system(rover_rankings)


        model.learn(total_timesteps=total_retrain_timesteps + exp_args.num_timesteps, progress_bar=True,
                    reset_num_timesteps=False)




if __name__ == "__main__":
    main(sys.argv)

