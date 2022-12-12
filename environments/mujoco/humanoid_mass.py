"""
    Based on environment in PEARL:
    https://github.com/katerakelly/oyster/blob/master/rlkit/envs/humanoid_dir.py
"""
import numpy as np
from gym.envs.mujoco import HumanoidEnv as HumanoidEnv
from gym import spaces

import random


class HumanoidMassEnv(HumanoidEnv):

    def __init__(self, max_episode_steps=200, eval_mode=False):
        self.task_dim = 1
        self._max_episode_steps = max_episode_steps
        self.action_scale = 1  # Mujoco environment initialization takes a step,
        super(HumanoidMassEnv, self).__init__()

        # Override action space to make it range from  (-1, 1)
        assert (self.action_space.low == -self.action_space.high).all()
        self.action_scale = self.action_space.high[0]
        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=self.action_space.shape)  # Overriding original action_space which is (-0.4, 0.4, shape = (17, ))

        # save original cheetah properties (tasks are defined as ratios of these)
        # self.original_acc = self.model.actuator_acc0.copy()
        self.original_mass = self.model.body_mass.copy()
        self.original_size = self.model.geom_size.copy()

        self.set_task(self.sample_task())

        self._time = 0
        self._return = 0
        self._last_return = 0
        self._curr_rets = []

    def step(self, action):
        observation, reward, done, done2, infos = super().step(action)
        if done:
            # not alive - keep running but remove alive-reward
            reward -= 5.0
            infos['reward_alive'] = 0
            done = False
        infos['task'] = self.get_task()
        self._time += 1
        self._return += reward
        if self._time % self._max_episode_steps == 0:
            self._last_return = self._return
            self._curr_rets.append(self._return)
            self._return = 0
        return observation, reward, done, infos

    def _get_obs(self):
        data = self.sim.data
        return np.concatenate([data.qpos.flat[2:],
                               data.qvel.flat,
                               data.cinert.flat,
                               data.cvel.flat,
                               data.qfrc_actuator.flat,
                               data.cfrc_ext.flat])

    def get_last_return(self):
        return np.sum(self._curr_rets)

    def set_task(self, task):
        self.task = task

        for i in range(len(self.model.body_mass)):
            self.model.body_mass[i] = task * self.original_mass[i]
        self.model.geom_size[1:, 0] = task * self.original_size[1:, 0]

        # # butt, feet, hands
        # for i_task, (i_size, i_mass) in enumerate(zip(([5], [8,11], [14,17]), ([3], [6,9], [11,13]))):
        #     # mass + width
        #     for i in i_mass:
        #         self.model.body_mass[i] = task[i_task] * self.original_mass[i]
        #     for i in i_size:
        #         self.model.geom_size[i, 0] = task[i_task] * self.original_size[i, 0]

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
