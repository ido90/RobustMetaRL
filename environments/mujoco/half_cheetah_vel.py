import random

import numpy as np

from gym.spaces import Box
from .half_cheetah import HalfCheetahEnv


class HalfCheetahVelEnv(HalfCheetahEnv):
    """Half-cheetah environment with target velocity, as described in [1]. The
    code is adapted from
    https://github.com/cbfinn/maml_rl/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/rllab/envs/mujoco/half_cheetah_env_rand.py

    The half-cheetah follows the dynamics from MuJoCo [2], and receives at each
    time step a reward composed of a control cost and a penalty equal to the
    difference between its current velocity and the target velocity. The tasks
    are generated by sampling the target velocities from the uniform
    distribution on [0, 2].

    [1] Chelsea Finn, Pieter Abbeel, Sergey Levine, "Model-Agnostic
        Meta-Learning for Fast Adaptation of Deep Networks", 2017
        (https://arxiv.org/abs/1703.03400)
    [2] Emanuel Todorov, Tom Erez, Yuval Tassa, "MuJoCo: A physics engine for
        model-based control", 2012
        (https://homes.cs.washington.edu/~todorov/papers/TodorovIROS12.pdf)
    """

    def __init__(self, max_episode_steps=200, eval_mode=False):
        self.eval_mode = eval_mode
        self.set_task(self.sample_tasks(1)[0])
        self._max_episode_steps = max_episode_steps
        self.task_dim = 1
        super(HalfCheetahVelEnv, self).__init__()
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(20,),
                                     dtype=np.float64)
        self._time = 0
        self._return = 0
        self._last_return = 0

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]

        forward_vel = (xposafter - xposbefore) / self.dt
        forward_reward = -1.0 * abs(forward_vel - self.goal_velocity)
        ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(action))

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        done = False
        infos = dict(reward_forward=forward_reward,
                     reward_ctrl=-ctrl_cost,
                     task=self.get_task())
        self._time += 1
        self._return += reward
        if self._time % self._max_episode_steps == 0:
            # print(f'[{self._time//self._max_episode_steps}] '
            #       f'{self.goal_velocity},\t{self._return}')
            self._last_return = self._return
            self._return = 0
        return observation, reward, done, infos

    def get_last_return(self):
        return self._last_return

    def set_task(self, task):
        if isinstance(task, np.ndarray):
            task = task[0]
        self.goal_velocity = task
        return task

    def get_task(self):
        return np.array([self.goal_velocity])

    def sample_task(self, p_orig=1, alpha=0.1):
        x = random.uniform(0.0, 1.0)
        if p_orig < 1:
            is_orig = random.random() < p_orig
            if not is_orig:
                x = (1-alpha) + alpha*x
        return 7.0 * x

    def sample_tasks(self, n_tasks):
        reg = 1  # set 1 for standard sampling, 0.25 for 0.75-tail
        p_orig = 1 if self.eval_mode else reg
        return [self.sample_task(p_orig) for _ in range(n_tasks)]

    def reset_task(self, task):
        if task is None:
            task = self.sample_tasks(1)[0]
        self.set_task(task)
        self._time = 0
        self._last_return = self._return
        self._return = 0
        # self.reset()


class HalfCheetahRandVelOracleEnv(HalfCheetahVelEnv):

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
            self.get_body_com("torso").flat,
            [self.goal_velocity]
        ])
