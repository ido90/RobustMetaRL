"""
Main scripts to start experiments.
Takes a flag --env-type (see below for choices) and loads the parameters from the respective config file.
"""
import argparse
import logging
import os
import warnings

import numpy as np
import torch
import wandb

# get configs
from config import args_khazad_dum_varibad
from config.mujoco import args_cheetah_vel_rl2, args_cheetah_vel_varibad, args_cheetah_mass_varibad, \
    args_cheetah_body_varibad,  args_ant_goal_rl2, args_ant_goal_varibad, args_ant_mass_varibad, \
    args_humanoid_vel_varibad, args_humanoid_mass_varibad, args_humanoid_body_varibad
from environments.parallel_envs import make_vec_envs
from learner import Learner
from metalearner import MetaLearner

def generate_exp_label(args):
    if not args.exp_label:
        if args.oracle:
            method = 'oracle'  # 'Oracle'
        elif args.cem == 0 and args.tail == 0:
            method = 'varibad'  # 'VariBAD'
        elif args.cem == 1:
            method = 'cembad'  # 'RoML'
        elif args.tail == 1:
            method = 'cvrbad'  # 'CVaR_MRL'
        else:
            raise ValueError(args.cem, args.tail)

        env_name_map = {
            'KhazadDum-v0':'kd',
            'HalfCheetahVel-v0':'hcv',
            'HalfCheetahMass-v0':'hcm',
            'HalfCheetahBody-v0':'hcb',
            'HumanoidVel-v0':'humv',
            'HumanoidMass-v0':'humm',
            'HumanoidBody-v0':'humb',
        }
        env_name = env_name_map[args.env_name]

        args.exp_label = f'{env_name}_{method}'

    try:
        if args.exp_suffix:
            args.exp_label = f'{args.exp_label}_{args.exp_suffix}'
    except:
        warnings.warn(f'Missing attribute args.exp_label')

    return args

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-type', default='gridworld_varibad')
    args, rest_args = parser.parse_known_args()
    env = args.env_type

    if env == 'khazad_dum_varibad':
        args = args_khazad_dum_varibad.get_args(rest_args)

    # --- MUJOCO ---

    # - Cheetah -
    elif env == 'cheetah_vel_varibad':
        args = args_cheetah_vel_varibad.get_args(rest_args)
    elif env == 'cheetah_vel_rl2':
        args = args_cheetah_vel_rl2.get_args(rest_args)
    elif env == 'cheetah_mass_varibad':
        args = args_cheetah_mass_varibad.get_args(rest_args)
    elif env == 'cheetah_body_varibad':
        args = args_cheetah_body_varibad.get_args(rest_args)
    #
    # - Humanoid -
    elif env == 'humanoid_vel_varibad':
        args = args_humanoid_vel_varibad.get_args(rest_args)
    elif env == 'humanoid_mass_varibad':
        args = args_humanoid_mass_varibad.get_args(rest_args)
    elif env == 'humanoid_body_varibad':
        args = args_humanoid_body_varibad.get_args(rest_args)
    else:
        raise Exception("Invalid Environment")

    args = generate_exp_label(args)

    # warning for deterministic execution
    if args.deterministic_execution:
        print('Envoking deterministic code execution.')
        if torch.backends.cudnn.enabled:
            warnings.warn('Running with deterministic CUDNN.')
        if args.num_processes > 1:
            raise RuntimeError('If you want fully deterministic code, run it with num_processes=1.'
                               'Warning: This will slow things down and might break A2C if '
                               'policy_num_steps < env._max_episode_steps.')

    # if we're normalising the actions, we have to make sure that the env expects actions within [-1, 1]
    if args.norm_actions_pre_sampling or args.norm_actions_post_sampling:
        envs = make_vec_envs(env_name=args.env_name, seed=0, num_processes=args.num_processes,
                             gamma=args.policy_gamma, device='cpu',
                             episodes_per_task=args.max_rollouts_per_task,
                             normalise_rew=args.norm_rew_for_policy, ret_rms=None,
                             tasks=None,
                             )
        assert np.unique(envs.action_space.low) == [-1]
        assert np.unique(envs.action_space.high) == [1]

    # clean up arguments
    if args.disable_metalearner or args.disable_decoder:
        args.decode_reward = False
        args.decode_state = False
        args.decode_task = False

    if hasattr(args, 'decode_only_past') and args.decode_only_past:
        args.split_batches_by_elbo = True
    # if hasattr(args, 'vae_subsample_decodes') and args.vae_subsample_decodes:
    #     args.split_batches_by_elbo = True

    # init wandb
    # (use mode="disabled" to disable wandb)
    if args.use_wandb:
        ngc_run = os.path.isdir('/ws')
        if ngc_run:
            ngc_dir = '/result/wandb/'  # args.ngc_path
            os.makedirs(ngc_dir, exist_ok=True)
            logging.info('NGC run detected. Setting path to workspace: {}'.format(ngc_dir))
            wandb.init(project="roml", sync_tensorboard=True, config=args, dir=ngc_dir)
        else:
            wandb.init(project="roml", sync_tensorboard=True, config=args)

    # begin training (loop through all passed seeds)
    seed_list = [args.seed] if isinstance(args.seed, int) else args.seed
    for seed in seed_list:
        print('training', seed)
        args.seed = seed
        args.action_space = None

        if args.disable_metalearner:
            # If `disable_metalearner` is true, the file `learner.py` will be used instead of `metalearner.py`.
            # This is a stripped down version without encoder, decoder, stochastic latent variables, etc.
            learner = Learner(args)
        else:
            learner = MetaLearner(args)
        learner.train()

        # test
        try:
            n_tasks = args.test_tasks
        except:
            n_tasks = 1000
        learner.test(n_tasks=n_tasks, load='final')
        learner.test(n_tasks=n_tasks, load='best_cvar')
        learner.test(n_tasks=n_tasks, load='best_mean')


if __name__ == '__main__':
    main()
