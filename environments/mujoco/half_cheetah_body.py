import random

import numpy as np

from gym.spaces import Box
from .half_cheetah import HalfCheetahEnv


class HalfCheetahBodyEnv(HalfCheetahEnv):
    """Half-cheetah environment with varying body. The code is adapted from
    https://github.com/lmzintgraf/varibad/blob/master/environments/mujoco/half_cheetah_vel.py

    The half-cheetah follows the dynamics and rewards from MuJoCo.
    Its tasks correspond to different values of cheetah body mass and TODO,
    sampled log-uniformly in the range of [50%, 200%] of their original values.
    """

    def __init__(self, max_episode_steps=200, eval_mode=False):
        self.eval_mode = eval_mode
        self.set_task(self.sample_task())
        self._max_episode_steps = max_episode_steps
        self.task_dim = 1
        super(HalfCheetahBodyEnv, self).__init__()
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(20,),
                                     dtype=np.float64)

        self.original_mass_vec = self.model.body_mass.copy()
        print(self.original_mass_vec)  # TODO TMP

        self._time = 0
        self._return = 0
        self._last_return = 0

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]

        forward_reward = (xposafter - xposbefore) / self.dt
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
            #       f'{self.task},\t{self._return}')
            self._last_return = self._return
            self._return = 0
        return observation, reward, done, infos

    def get_last_return(self):
        return self._last_return

    def set_task(self, task):
        if isinstance(task, np.ndarray):
            task = task[0]
        self.task = task
        for i in range(len(self.model.body_mass)):
            self.model.body_mass[i] = task * self.original_mass_vec[i]
        print(task)
        print(self.model.body_mass)  # TODO TMP
        return task

    def get_task(self):
        return self.task

    def sample_task(self):
        return 2 ** random.uniform(-1, 1)

    def sample_tasks(self, n_tasks):
        return [self.sample_task() for _ in range(n_tasks)]

    def reset_task(self, task):
        if task is None:
            task = self.sample_task()
        self.set_task(task)
        self._time = 0
        self._last_return = self._return
        self._return = 0
        # self.reset()
