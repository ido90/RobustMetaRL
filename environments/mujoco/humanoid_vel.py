"""
    Based on environment in PEARL:
    https://github.com/katerakelly/oyster/blob/master/rlkit/envs/humanoid_dir.py
"""
import numpy as np
from gym.envs.mujoco import HumanoidEnv as HumanoidEnv
from gym import spaces

import random


class HumanoidVelEnv(HumanoidEnv):

    def __init__(self, max_episode_steps=200, eval_mode=False):
        self.task_dim = 1
        self._max_episode_steps = max_episode_steps
        self.action_scale = 1  # Mujoco environment initialization takes a step,
        super(HumanoidVelEnv, self).__init__()

        # Override action space to make it range from  (-1, 1)
        assert (self.action_space.low == -self.action_space.high).all()
        self.action_scale = self.action_space.high[0]
        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=self.action_space.shape)  # Overriding original action_space which is (-0.4, 0.4, shape = (17, ))

        self.goal_velocity = None
        self.set_task(self.sample_task())

        self._time = 0
        self._return = 0
        self._last_return = 0
        self._curr_rets = []

    def step(self, action):
        observation, reward, done, done2, infos = super().step(action)

        # if not alive - don't cut episode, but do remove alive-reward
        if done:
            reward -= 5.0
            infos['reward_alive'] = 0
            done = False

        # replace velocity reward with velocity-difference reward
        vel_reward = infos['reward_linvel']
        vel = vel_reward / 1.25
        reward -= vel_reward
        reward += -1.25 * abs(vel - self.goal_velocity)

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
        if isinstance(task, np.ndarray):
            task = task[0]
        self.goal_velocity = task
        return task

    def get_task(self):
        return np.array([self.goal_velocity])

    def sample_task(self):
        x = random.uniform(0.0, 1.0)
        return 2.5 * x

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
