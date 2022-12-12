import random

import numpy as np

from gym.spaces import Box
from .half_cheetah import HalfCheetahEnv


class HalfCheetahBodyEnv(HalfCheetahEnv):
    """Half-cheetah environment with varying body. The code is adapted from
    https://github.com/lmzintgraf/varibad/blob/master/environments/mujoco/half_cheetah_vel.py

    The half-cheetah follows the dynamics and rewards from MuJoCo.
    Its tasks correspond to different values of cheetah body mass, damping and torso length,
    sampled log-uniformly in the range of [50%, 200%] of their original values.
    """

    def __init__(self, max_episode_steps=200, eval_mode=False):
        self.eval_mode = eval_mode
        self._max_episode_steps = max_episode_steps
        self.task_dim = 3
        super(HalfCheetahBodyEnv, self).__init__()
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(20,),
                                     dtype=np.float64)

        # save original cheetah properties (tasks are defined as ratios of these)
        self.original_mass_vec = self.model.body_mass.copy()  # 8 elements
        # self.original_inertia_vec = self.model.body_inertia.copy()  # 8x3 elements
        self.original_damp_vec = self.model.dof_damping.copy()  # 9 elements
        self.original_len = self.model.geom_size[2, 1].copy()  # 1 element

        self.set_task(self.sample_task())

        self._time = 0
        self._return = 0
        self._last_return = 0
        self._curr_rets = []

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]

        forward_reward = (xposafter - xposbefore) / self.dt
        ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(action))

        observation = self._get_obs()
        # this value is in [-0.1,0.1] when the cheetah is straight, and in [-0.6,-0.4] when it's upside-down.
        #  we penalize the cheetah being upside down.
        reward_height = observation[0]
        reward = forward_reward - ctrl_cost + reward_height
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
            self._curr_rets.append(self._return)
            self._return = 0
        return observation, reward, done, infos

    def get_last_return(self):
        return np.sum(self._curr_rets)

    def set_task(self, task):
        self.task = task

        self.model.geom_size[1:, 0] = 0.046 * task[0]  # increase thickness with mass
        for i in range(len(self.model.body_mass)):
            self.model.body_mass[i] = task[0] * self.original_mass_vec[i]
        # for i in range(len(self.model.body_inertia)):
        #     self.model.body_inertia[i] = task[1] * self.original_inertia_vec[i]
        for i in range(len(self.model.dof_damping)):
            self.model.dof_damping[i] = task[1] * self.original_damp_vec[i]
        self.model.geom_size[2, 1] = task[2] * self.original_len

        return task

    def get_task(self):
        return self.task

    def sample_task(self):
        return np.array([2 ** random.uniform(-1, 1)
                         for _ in range(self.task_dim)])

    def sample_tasks(self, n_tasks):
        return [self.sample_task() for _ in range(n_tasks)]

    def reset_task(self, task):
        if task is None:
            task = self.sample_task()
        self.set_task(task)
        self._time = 0
        self._last_return = self._return
        self._curr_rets = []
        self._return = 0
        # self.reset()
