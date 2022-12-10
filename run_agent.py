import argparse
import os, time
import warnings
import matplotlib.pyplot as plt
import numpy as np
import torch
import imageio
from environments.parallel_envs import make_vec_envs
from utils import helpers as utl
import analysis
from config import args_khazad_dum_varibad
from config.mujoco import \
    args_cheetah_vel_varibad, args_cheetah_mass_varibad, args_cheetah_body_varibad, \
    args_humanoid_mass_varibad, args_humanoid_body_varibad
from metalearner import MetaLearner

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def run_agent(learner, tasks=None, num_episodes=1, gif='demo', dur=1, sleep=0.05):
    # GIF args
    dur = 0.01 * dur

    ret_rms = learner.envs.venv.ret_rms if learner.args.norm_rew_for_policy else None
    args = learner.args
    policy = learner.policy
    encoder = learner.vae.encoder
    iter_idx = learner.args.seed

    env_name = args.env_name
    if hasattr(args, 'test_env_name'):
        env_name = args.test_env_name
    if num_episodes is None:
        num_episodes = args.max_rollouts_per_task
    num_processes = args.num_processes

    # --- set up the things we want to log ---

    # for each process, we log the returns during the first, second, ... episode
    # (such that we have a minimum of [num_episodes]; the last column is for
    #  any overflow and will be discarded at the end, because we need to wait until
    #  all processes have at least [num_episodes] many episodes)
    returns_per_episode = torch.zeros((num_processes, num_episodes + 1)).to(device)

    # --- initialise environments and latents ---

    envs = make_vec_envs(env_name,
                         seed=args.seed * 42 + iter_idx,
                         num_processes=num_processes,
                         gamma=args.policy_gamma,
                         device=device,
                         rank_offset=num_processes + 1,  # to use diff tmp folders than main processes
                         episodes_per_task=num_episodes,
                         normalise_rew=args.norm_rew_for_policy,
                         ret_rms=ret_rms,
                         tasks=tasks,
                         add_done_info=args.max_rollouts_per_task > 1,
                         eval_mode=True,
                         )
    num_steps = envs._max_episode_steps

    # reset environments
    state, belief, task = utl.reset_env(envs, args)
    tasks = envs.get_task()
    print('Tasks:')
    print(tasks)

    # this counts how often an agent has done the same task already
    task_count = torch.zeros(num_processes).long().to(device)

    if encoder is not None:
        # reset latent state to prior
        latent_sample, latent_mean, latent_logvar, hidden_state = encoder.prior(num_processes)
    else:
        latent_sample = latent_mean = latent_logvar = hidden_state = None

    states = []
    for episode_idx in range(num_episodes):

        ep_obs = []

        for step_idx in range(num_steps):

            with torch.no_grad():
                _, action = utl.select_action(args=args,
                                              policy=policy,
                                              state=state,
                                              belief=belief,
                                              task=task,
                                              latent_sample=latent_sample,
                                              latent_mean=latent_mean,
                                              latent_logvar=latent_logvar,
                                              deterministic=True)

            # observe reward and next obs
            [state, belief, task], (rew_raw, rew_normalised), done, infos = utl.env_step(envs, action, args)
            states.append(state.reshape(-1).cpu().numpy())
            done_mdp = [info['done_mdp'] for info in infos]

            if encoder is not None:
                # update the hidden state
                latent_sample, latent_mean, latent_logvar, hidden_state = utl.update_encoding(encoder=encoder,
                                                                                              next_obs=state,
                                                                                              action=action,
                                                                                              reward=rew_raw,
                                                                                              done=None,
                                                                                              hidden_state=hidden_state)

            # add rewards
            returns_per_episode[range(num_processes), task_count] += rew_raw.view(-1)

            for i in np.argwhere(done_mdp).flatten():
                # count task up, but cap at num_episodes + 1
                task_count[i] = min(task_count[i] + 1, num_episodes)  # zero-indexed, so no +1
            if np.sum(done) > 0:
                done_indices = np.argwhere(done.flatten()).flatten()
                state, belief, task = utl.reset_env(envs, args, indices=done_indices, state=state)

            # update GIF
            if 'Khazad' not in env_name:
                if gif:
                    env = envs.venv.unwrapped.envs[0].unwrapped
                    obs = env.render(mode='rgb_array')  # , width=width, height=height)
                    ep_obs.append(obs)
                elif sleep:
                    env = envs.venv.unwrapped.envs[0].unwrapped
                    env.render()
                    time.sleep(sleep)

        # save GIF
        if gif:
            fname = gif
            if num_episodes > 1:
                fname += f'_{episode_idx}'
            fname = f'logs/gifs/{fname}'
            print(f'Saving {fname}')

            if 'Khazad' in env_name:
                env = envs.venv.unwrapped.envs[0].unwrapped
                env.show_state()
                time.sleep(0.5)
                plt.savefig(f'{fname}.png', bbox_inches='tight')
            else:
                with imageio.get_writer(f'{fname}.gif', mode='I', duration=dur) as writer:
                    for obs_np in ep_obs:
                        writer.append_data(obs_np)
                # also save trajectory of states
                np.save(f'{fname}.npy', np.stack(states))

    returns_per_episode = returns_per_episode[:, :num_episodes]
    print('Returns:')
    print(returns_per_episode.cpu().numpy())

    envs.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-type', default='cheetah_vel_varibad')
    args, rest_args = parser.parse_known_args()
    env = args.env_type

    if env == 'cheetah_vel_varibad':
        args = args_cheetah_vel_varibad.get_args(rest_args)
    elif env == 'cheetah_mass_varibad':
        args = args_cheetah_mass_varibad.get_args(rest_args)
    elif env == 'cheetah_body_varibad':
        args = args_cheetah_body_varibad.get_args(rest_args)
    elif env == 'humanoid_mass_varibad':
        args = args_humanoid_mass_varibad.get_args(rest_args)
    elif env == 'humanoid_body_varibad':
        args = args_humanoid_body_varibad.get_args(rest_args)
    elif env == 'khazad_dum_varibad':
        args = args_khazad_dum_varibad.get_args(rest_args)
    else:
        raise Exception("Invalid Environment")

    short_name = dict(cheetah_vel_varibad='hcv', cheetah_mass_varibad='hcm',
                      cheetah_body_varibad='hcb', humanoid_body_varibad='humb',
                      humanoid_mass_varibad='humm', khazad_dum_varibad='kd')[env]
    method = 'varibad'
    if args.cem:
        if args.oracle:
            method = 'oracbad'
        elif args.cem == 1:
            method = 'cembad'
        else:
            method = 'cesbad'
    elif args.tail == 1:
        method = 'cvrbad'
    elif args.tail == 2:
        method = 'schedbad'

    tasks = None
    if isinstance(args.alpha, (tuple, list)):
        tasks = [args.alpha]

    save_gif = args.save_interval > 0
    num_episodes = args.max_rollouts_per_task

    # set args for demo run
    args.deterministic_execution = True
    args.num_processes = 1
    args.results_log_dir = 'tmp_logs'

    # warning for deterministic execution
    if args.deterministic_execution:
        print('Envoking deterministic code execution.')
        if torch.backends.cudnn.enabled:
            warnings.warn('Running with deterministic CUDNN.')

    # clean up arguments
    if args.disable_metalearner or args.disable_decoder:
        args.decode_reward = False
        args.decode_state = False
        args.decode_task = False

    if hasattr(args, 'decode_only_past') and args.decode_only_past:
        args.split_batches_by_elbo = True

    # begin training (loop through all passed seeds)
    seed_list = [args.seed] if isinstance(args.seed, int) else args.seed
    for seed in seed_list:
        print('running', seed)
        args.seed = seed
        args.action_space = None
        base_path = analysis.get_base_path(args.env_name)
        dir = analysis.get_dir(base_path, short_name, method, args.seed)
        # pth = f'{base_path}/{dir}/final_models'
        # pth = f'{base_path}/{dir}/best_models'
        # pth = f'{base_path}/{dir}/best_mean_models'
        pth = f'{base_path}/{dir}/best_cvar_models'
        print('\nLoading model from:', pth)
        learner = MetaLearner(args)
        learner.load_model(save_path=pth)
        run_agent(
            learner,
            tasks=tasks,
            gif=f'{short_name}_{method}_{args.seed}' if save_gif else None,
            dur=1 if short_name.startswith('hum') else 5,
            num_episodes=num_episodes
        )


if __name__ == '__main__':
    main()
