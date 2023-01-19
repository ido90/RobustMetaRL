import random

import numpy as np

from environments.mujoco.ant import AntEnv


class AntMassEnv(AntEnv):
    """
    Forward/backward ant direction environment
    """

    def __init__(self, max_episode_steps=200):
        self._max_episode_steps = max_episode_steps
        self.task_dim = 1
        self.set_task(self.sample_tasks(1)[0])
        super(AntMassEnv, self).__init__()

        # save original cheetah properties (tasks are defined as ratios of these)
        # self.original_acc = self.model.actuator_acc0.copy()
        self.original_mass = self.model.body_mass.copy()
        # self.original_size = self.model.geom_size.copy()

        self._time = 0
        self._return = 0
        self._last_return = 0
        self._curr_rets = []

    def step(self, action):
        torso_xyz_before = np.array(self.get_body_com("torso"))
        self.do_simulation(action, self.frame_skip)
        torso_xyz_after = np.array(self.get_body_com("torso"))
        torso_velocity = torso_xyz_after - torso_xyz_before
        forward_reward = torso_velocity[0] / self.dt

        ctrl_cost = .5 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
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
        return np.array([2 ** random.uniform(-1, 1)
                         for _ in range(self.task_dim)])

    def sample_tasks(self, n_tasks):
        return [self.sample_task() for _ in range(n_tasks)]

    def set_task(self, task):
        self.task = task
        for i in range(len(self.model.body_mass)):
            self.model.body_mass[i] = task * self.original_mass[i]
        return task

    def get_task(self):
        return self.task

    def reset_task(self, task):
        if task is None:
            task = self.sample_task()
        self.set_task(task)
        self._time = 0
        self._last_return = self._return
        self._curr_rets = []
        self._return = 0
