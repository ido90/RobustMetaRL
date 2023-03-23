import random

import numpy as np

from environments.mujoco.ant import AntEnv


class AntVelEnv(AntEnv):
    """
    Forward/backward ant direction environment
    """

    def __init__(self, max_episode_steps=200):
        self._max_episode_steps = max_episode_steps
        self.task_dim = 1
        self.goal_velocity = 0
        self._time = 0
        self._return = 0
        self._last_return = 0
        self._curr_rets = []
        super(AntVelEnv, self).__init__()

        self.goal_velocity = None
        self.set_task(self.sample_tasks(1)[0])

    def step(self, action):
        torso_xyz_before = np.array(self.get_body_com("torso"))
        self.do_simulation(action, self.frame_skip)
        torso_xyz_after = np.array(self.get_body_com("torso"))
        torso_velocity = (torso_xyz_after - torso_xyz_before) / self.dt
        forward_reward = -1.25 * abs(torso_velocity[0] - self.goal_velocity)

        state = self.state_vector()
        survive = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        ctrl_cost = .5 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0 if survive else 0.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        done = False
        ob = self._get_obs()

        self._time += 1
        self._return += reward
        if self._time % self._max_episode_steps == 0:
            self._last_return = self._return
            self._curr_rets.append(self._return)
            self._return = 0

        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            torso_velocity=torso_velocity,
            task=self.get_task()
        )

    def get_last_return(self):
        return np.sum(self._curr_rets)

    def sample_task(self):
        x = random.uniform(0.0, 1.0)
        return 3 * x

    def sample_tasks(self, n_tasks):
        return [self.sample_task() for _ in range(n_tasks)]

    def set_task(self, task):
        if isinstance(task, np.ndarray):
            task = task[0]
        self.goal_velocity = task
        return task

    def get_task(self):
        return np.array([self.goal_velocity])

    def reset_task(self, task):
        if task is None:
            task = self.sample_task()
        self.set_task(task)
        self._time = 0
        self._last_return = self._return
        self._curr_rets = []
        self._return = 0
