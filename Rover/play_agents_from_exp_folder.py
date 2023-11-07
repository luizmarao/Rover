import os
import sys
import time

import numpy as np
import datetime
from glob import glob
from matplotlib import pyplot as plt

from Rover.utils.logger import safe_print
from Rover.algos.ppo.ppo import PPO_Rover
from Rover.utils.env_register import register_rover_environments
from Rover.utils.env_util import make_vec_env
from Rover.utils.arg_parser import common_arg_parser, parse_unknown_args, play_only_arg_parser
from Rover.utils.field_plot_tools import multicolor_line_plot

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

def generate_field_plots(obs, act, net, env, background=None, goal_mode=-1, savepath=None, show=False):
    """
    Generate special plots, with the rover trajectory over the field image as background.
    The line color changes according to the parameter sent as line_intensity, and a band is drawn around it with
    another variable.
    :param obs: [gps_sensor.flat, orientation_rover.flat, ghost_steer_angle, speed_sensor.flat, rover_ang_speed,
             ghost_steer_angspeed, coordinates_goal.flat, goal.flat]
    :param act: [steering_motor, acc_motor]
    :param net: number/name of the network that generated the data
    :param env: name of the environment where the data was acquired
    :param background: picture of the field where the simulation occurred
    :param goal_mode: -1 if full run, or the number of specific goal
    :param savepath: the experiment's path, or any other where the folder with the images will be created
    :param show: whether to show or not the plots generated
    :return:
    """

    steering_motor = act[0]
    acc = act[1]
    x = obs[0]
    y = obs[1]
    steering_angle = obs[4]
    vx = obs[5]
    vy = obs[6]
    speed = np.hypot(vx, vy)

    plot_modes = ['Acc+SteeringSignals', 'Speed+SteeringAngle']

    if savepath is not None:
        plots_folder = os.path.join(savepath, 'field_plots')
        os.makedirs(plots_folder, exist_ok=True)
        date_string = datetime.datetime.now().strftime("-%Y-%m-%d-%H-%M-%S")
        short_env_name = env.split('-')[0][5:]
        if goal_mode == -1:
            goal_string = 'full_mode'
        else:
            goal_string = f"g{goal_mode}_mode"

        plots_paths = [os.path.join(plots_folder, f"{short_env_name}_{net}_{goal_string}_{mode}{date_string}.png")
                       for mode in plot_modes]
    else:
        plots_paths = [None] * len(plot_modes)

    titles = ['Acceleration and Steering signals over the path', 'Rover Speed and Steering Angle over the path']
    bar_labels = ['Acceleration Signal', 'Rover Speed (m/s)']

    from matplotlib.colors import LinearSegmentedColormap
    colors = ['mistyrose', 'red', 'darkred']
    nodes = [0.0, 0.85, 1.0]
    cmap = LinearSegmentedColormap.from_list('growingred', list(zip(nodes, colors)))

    multicolor_line_plot(x=x, y=y, line_intensity=acc, band=steering_motor, band_size_limit=1, background=background,
                         cmap='coolwarm', facecolor='black', edgecolor='white', title=titles[0],
                         colorbar_label=bar_labels[0], savepath=plots_paths[0], show=show)
    multicolor_line_plot(x=x, y=y, line_intensity=speed, band=steering_angle, band_size_limit=3, background=background,
                         cmap=cmap, facecolor='black', edgecolor='white', title=titles[1],
                         colorbar_label=bar_labels[1], savepath=plots_paths[1], show=show)
    time.sleep(1.0)  # to avoid plots with the same timestamp



def main(args):
    args_ = args
    exp_arg_parser = common_arg_parser()
    play_arg_parser = play_only_arg_parser()
    play_args, unknown_play_args = play_arg_parser.parse_known_args(args)
    extra_play_args = parse_cmdline_kwargs(unknown_play_args)

    device = play_args.device

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
    if isinstance(goals, str):
        goals = eval(goals)

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

    if play_args.export_plot and rover_env.startswith('Rover4W-'):  # Always the same img, don't need to read each eps
        bg = plt.imread('utils/Rover4W_field.png')
    else:
        bg = None
    # PLAY
    error_loading = []
    for net in nets:
        obs_to_plot = [[] for i in range(num_environments)]
        actions_to_plot = [[] for i in range(num_environments)]
        net_num = net.split('/')[-1].split('.')[0]
        try:
            model = PPO_Rover.load(path=net, device=device)
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
            envs.seed()
            observations = envs.reset()
            episode_counts = np.zeros(num_environments, dtype="int")

            if num_play_eps == -1:
                num_play_eps = np.inf
            episode_count_targets = np.array([(num_play_eps + i) // num_environments for i in range(num_environments)],
                                             dtype="int")
            while (episode_counts < episode_count_targets).any():
                actions, _ = model.predict(observation=observations, deterministic=True)
                if play_args.export_plot:  # store obs and actions to future plot
                    split_obs = np.vsplit(observations[:, 0:14], num_environments)
                    split_actions = np.vsplit(actions, num_environments)
                    for i in range(num_environments):
                        obs_to_plot[i].append(split_obs[i].T)
                        actions_to_plot[i].append(split_actions[i].T)
                observations, rewards, dones, infos = envs.step(actions)
                envs.render()
                for i in range(num_environments):
                    if episode_counts[i] < episode_count_targets[i]:
                        if dones[i]:
                            episode_counts[i] += 1
                            if play_args.export_plot:  # make the arrays, plot and clean lists
                                obs = np.hstack(obs_to_plot[i])
                                act = np.hstack(actions_to_plot[i])
                                if not rover_env.startswith('Rover4W-'):  # needs to load background image
                                    bgs = envs.env_method('camera_rendering', only_overview=True)
                                    bg = bgs[i][18:-19, 15:-16, :]  # crop image's black borders
                                generate_field_plots(obs=obs, act=act, net=net_num, env=rover_env, background=bg,
                                                     goal_mode=goal, savepath=play_exp_dir, show=False)
                                obs_to_plot[i].clear()
                                actions_to_plot[i].clear()

    if len(error_loading) > 0:
        safe_print('Network(s) {} skipped due to loading error.'.format(', '.join(error_loading)))
    envs.close()
    print()


if __name__ == "__main__":
    main(sys.argv)

