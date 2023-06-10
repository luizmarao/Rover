import os
import sys
import numpy as np

from Rover.algos.ppo.ppo import PPO_Rover
from Rover.utils.env_register import register_rover_environments
from Rover.utils.env_util import make_vec_env
from Rover.utils.logger import configure
from Rover.utils.lr_schedulers import linear_schedule
from Rover.utils.networks_ranking import RoverRankingSystem
from Rover.utils.arg_parser import common_arg_parser, parse_unknown_args
from Rover.utils.networks import RovernetClassic

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
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)

    os.environ["MUJOCO_GL"] = 'egl'  # Set mujoco rendering to dedicated GPU
    EXPERIMENT_NAME = args.exp_name  # ns=num steps, bs=batch size, e=epochs, gr=goal_rwd
    os.environ["SB3_LOGDIR"] = os.path.expanduser(args.exp_root_folder)
    assert isinstance(os.environ["SB3_LOGDIR"], str)
    os.makedirs(os.environ["SB3_LOGDIR"], exist_ok=True)

    # set up logger
    logger = configure(folder=None, format_strings=["stdout", "csv", "tensorboard"], exp_name=EXPERIMENT_NAME)

    ## ENVIRONMENT PARAMETERS ##
    rover_env = args.env
    num_environments = args.num_env
    env_kwargs = {"render_mode": "rgb_array", 'gamma': args.gamma, 'start_at_initpos': args.start_at_initpos,
                  'end_after_current_goal': args.end_after_current_goal, 'random_current_goal': args.random_current_goal}
    env_kwargs.update(extra_args)
    monitor_kwargs = dict(info_keywords=['death', 'goal_reached_flag', 'timeout'], reset_keywords=['current_goal'])

    ## NETWORK STRUCTURE PARAMETERS ##
    net_arch = args.net_arch

    ## PPO PARAMETERS ##
    total_learning_timesteps = int(args.num_timesteps)
    n_steps = args.n_steps  # for each env per update
    seed = args.seed
    if args.lr_schedule:
        learning_rate = linear_schedule(args.lr)
    else:
        learning_rate = args.lr
    gamma = args.gamma
    n_epochs = args.n_epochs  # networks training epochs
    gae_lambda = args.gae_lambda
    assert args.batch_size <= n_steps * num_environments, "batch_size must not exceed n_steps * num_environments!"
    batch_size = args.batch_size  # was 64 in default algo
    clip_range = args.clip_range
    normalize_advantage = True
    ent_coef = args.ent_coef
    max_grad_norm = args.max_grad_norm
    use_sde = args.use_sde
    target_kl = args.target_kl
    clear_ep_info_buffer_every_iteration = True

    network_type = "MlpPolicy"
    policy_kwargs = dict(
        net_arch=net_arch,
    )
    if rover_env.startswith('Rover4WeDoubleCameraPacked'):
        rover_2cam_and_packed_images = True
    else:
        rover_2cam_and_packed_images = False

    if rover_env.startswith('Rover4We'):
        env_kwargs.update({"width": 440, "height": 270, "save_images": args.save_images, "verbose": args.env_verbose,
                  "img_reduced_size": args.img_red_size})
        if not 'Encoded' in rover_env:
            if args.networks_architecture == 'RovernetClassic':
                network_type = "CnnPolicy"
                policy_kwargs.update({'features_extractor_class': RovernetClassic,
                                      'share_features_extractor': args.share_features_extractor,
                                      'normalize_images': args.normalize_images,
                                      'features_extractor_kwargs': {
                                          'img_red_size': args.img_red_size,
                                          'rover_2cam_and_packed_images': rover_2cam_and_packed_images,
                                          'dynamic_obs_size': args.dynamic_obs_size,
                                          'conv_layers': args.conv_layers,
                                          'lin_layers': args.features_extractor_lin_layers,
                                      }
                                      })
        else:
            assert args.encoder_name is not None, 'This environment MUST receive a proper encoder name to work'
            env_kwargs.update({'encoder_name': args.encoder_name})

    envs = make_vec_env(rover_env, env_kwargs=env_kwargs, n_envs=num_environments, monitor_kwargs=monitor_kwargs)
    model = PPO_Rover(network_type, envs, policy_kwargs=policy_kwargs, verbose=1,
                      n_steps=n_steps,
                      learning_rate=learning_rate,
                      gamma=gamma,
                      n_epochs=n_epochs,
                      gae_lambda=gae_lambda,
                      batch_size=batch_size,
                      clip_range=clip_range,
                      normalize_advantage=normalize_advantage,
                      ent_coef=ent_coef,
                      max_grad_norm=max_grad_norm,
                      use_sde=use_sde,
                      target_kl=target_kl,
                      seed=seed,
                      clear_ep_info_buffer_every_iteration=clear_ep_info_buffer_every_iteration
                      )
    rover_rankings = RoverRankingSystem(networks_limit_per_ranking=args.networks_limit_per_ranking,
                                        num_rankings=args.num_rankings, save_path=logger.get_dir(),
                                        networks_subfolder='saved_networks', verbose=0)
    model.set_ranking_system(rover_rankings)
    model.set_logger(logger)
    model.learn(total_timesteps=total_learning_timesteps, progress_bar=True)

    # model.save(path=os.path.join(logger.get_dir(), "saved_model"), include=None, exclude=None)

    if args.play:
        pass



if __name__ == "__main__":
    main(sys.argv)

