def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def common_arg_parser():
    """
    Create an argparse.ArgumentParser for run.py.
    """

    def none_or_float(value):
        if value == 'None':
            return None
        return value
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='Rover4We-v2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=None)
    parser.add_argument('--alg', help='Algorithm', type=str, default='ppo2')  # Future change might include recurrent PPO
    parser.add_argument('--activation_fn', help='MLP activation function', type=str, default='nn.Tanh')
    parser.add_argument('--features_extractor_activation_fn', help='MLP activation function', type=str, default='nn.ReLU')
    parser.add_argument('--exp_name', help='The experiment name to go on the saving folder', type=str, default=None)
    parser.add_argument('--exp_root_folder', help='Directory in which the exp folder will be created to save logs and trained models', default=None, type=str)
    parser.add_argument('--num_timesteps', help='total learning steps', type=int, default=1e7),
    parser.add_argument('--n_steps', help='Steps for each env per iteration', type=int, default=4096),
    parser.add_argument('--n_epochs', help='Amount of training epochs for each iteration', type=int, default=50),
    parser.add_argument('--batch_size',  type=int, default=8192),
    parser.add_argument('--net_arch', help='network\'s mlp layers', default=dict(pi=[64, 64], vf=[64, 64]))
    parser.add_argument('--conv_layers', help='network\'s mlp layers', default=[(16, 8, 2), (32, 4, 2), (64, 2, 1)])
    parser.add_argument('--num_env', help='Number of environment copies being run in parallel.', default=4, type=int)
    parser.add_argument('--lr', help='Networks\' learning rate', default=0.00005, type=float)
    parser.add_argument('--no_lr_schedule', help='Use a linear schedule on the learning rate', default=False, action='store_true')
    parser.add_argument('--exp_lr_schedule', help='Use a exponential schedule on the learning rate', default=False,
                        action='store_true')
    parser.add_argument('--quad_lr_schedule', help='Use a quadratic schedule on the learning rate', default=False,
                        action='store_true')
    parser.add_argument('--exp_lr_decay', help='Decay rate of the exponential schedule on the learning rate',
                        default=0.9995, type=float)
    parser.add_argument('--use_sde', default=False, action='store_true')
    parser.add_argument('--gamma', type=float, default=0.99),
    parser.add_argument('--gae_lambda', type=float, default=0.95),
    parser.add_argument('--clip_range', type=float, default=0.08),
    parser.add_argument('--clip_range_vf', type=float, default=0.08),
    parser.add_argument('--ent_coef', type=float, default=0.1),
    parser.add_argument('--max_grad_norm', type=none_or_float, default=0.5),
    parser.add_argument('--target_kl', type=float, default=0.08),
    parser.add_argument('--networks_limit_per_ranking', type=int, default=5),
    parser.add_argument('--num_rankings', type=int, default=3),
    parser.add_argument('--dynamic_obs_size', type=int, default=14),
    parser.add_argument('--img_red_size', default=(64, 64)),
    parser.add_argument('--networks_architecture', type=str, default=None)
    parser.add_argument('--features_extractor_class', type=str, default=None)
    parser.add_argument('--features_extractor_lin_layers', type=str, default=None)
    parser.add_argument('--encoder_name', type=str, default=None)
    parser.add_argument('--dont_share_features_extractor', default=False, action='store_true')
    parser.add_argument('--dont_normalize_advantages', default=False, action='store_true')
    parser.add_argument('--start_at_initpos', default=False, action='store_true')
    parser.add_argument('--dont_end_after_current_goal', default=False, action='store_true')
    parser.add_argument('--dont_random_current_goal', default=False, action='store_true')
    parser.add_argument('--save_images', default=False, action='store_true')
    parser.add_argument('--normalize_images', default=False, action='store_true')
    parser.add_argument('--env_verbose', type=int, default=0),


    parser.add_argument('--play', default=False, action='store_true')
    parser.add_argument('--eval_goals', default=[-1, 1, 2])
    parser.add_argument('--num_eval_eps', type=int, default=200)
    parser.add_argument('--eval_exp_dir', type=str, default=None)
    return parser

def eval_only_arg_parser():
    """
    Create an argparse.ArgumentParser for run.py.
    """
    parser = arg_parser()
    parser.add_argument('--eval_goals', default=[-1, 1, 2])
    parser.add_argument('--num_eval_eps', type=int, default=200)
    parser.add_argument('--eval_exp_dir', type=str, default=None)
    parser.add_argument('--non_deterministic', default=False, action='store_true')
    parser.add_argument('--play', default=False, action='store_true')
    return parser

def retrain_arg_parser():
    """
    Create an argparse.ArgumentParser for run.py.
    """
    import os
    parser = arg_parser()
    parser.add_argument('--exp_root_folder', default=os.getcwd(), type=str)
    parser.add_argument('--num_timesteps', help='total learning steps', type=int, default=1e7),
    return parser

def play_only_arg_parser():
    """
    Create an argparse.ArgumentParser for run.py.
    """
    parser = arg_parser()
    parser.add_argument('--play_goals', default=[-1, 1, 2])
    parser.add_argument('--num_play_eps', help='If -1, play indefinitely (Thus, only one goal and agent will run)'
                        , type=int, default=200)
    parser.add_argument('--play_path', type=str, default=None)
    parser.add_argument('--export_plot', default=False, action='store_true')
    parser.add_argument('--dont_render', default=False, action='store_true')
    parser.add_argument('--device',  type=str, default='auto')
    return parser

def parse_unknown_args(args):
    """
    Parse arguments not consumed by arg parser into a dictionary
    """
    retval = {}
    preceded_by_key = False
    for arg in args:
        if arg.startswith('--'):
            if '=' in arg:
                key = arg.split('=')[0][2:]
                value = arg.split('=')[1]
                retval[key] = value
            else:
                key = arg[2:]
                preceded_by_key = True
        elif preceded_by_key:
            retval[key] = arg
            preceded_by_key = False

    return retval