import random

import numpy as np

from environments.mujoco.ant import AntEnv


class AntGoalEnv(AntEnv):
    def __init__(self, max_episode_steps=200, eval_mode=False):
        self._max_episode_steps = max_episode_steps
        self.task_dim = 2
        self.set_task(self.sample_tasks(1)[0])
        super(AntGoalEnv, self).__init__()

        self._time = 0
        self._return = 0
        self._last_return = 0
        self._curr_rets = []

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        xposafter = np.array(self.get_body_com("torso"))

        goal_reward = -np.sum(np.abs(xposafter[:2] - self.goal_pos))  # make it happy, not suicidal

        ctrl_cost = .1 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 0.0
        reward = goal_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        done = False
        ob = self._get_obs()

        self._time += 1
        self._return += reward
        if self._time % self._max_episode_steps == 0:
            self._last_return = self._return
            self._curr_rets.append(self._return)
            self._return = 0

        return ob, reward, done, dict(
            goal_forward=goal_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            task=self.get_task()
        )

    def get_last_return(self):
        return np.sum(self._curr_rets)

    def sample_tasks(self, num_tasks):
        a = np.array([random.random() for _ in range(num_tasks)]) * 2 * np.pi
        r = 5 * np.array([random.random() for _ in range(num_tasks)]) ** 0.5
        return np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)

    def sample_task(self):
        return self.sample_tasks(1)

    def set_task(self, task):
        self.goal_pos = task

    def get_task(self):
        return np.array(self.goal_pos)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

    def reset_task(self, task):
        if task is None:
            task = self.sample_task()
        self.set_task(task)
        self._time = 0
        self._last_return = self._return
        self._curr_rets = []
        self._return = 0


class AntGoalOracleEnv(AntGoalEnv):
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            self.goal_pos,
        ])