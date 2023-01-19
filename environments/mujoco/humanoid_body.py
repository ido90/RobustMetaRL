"""
    Based on environment in PEARL:
    https://github.com/katerakelly/oyster/blob/master/rlkit/envs/humanoid_dir.py
"""
import numpy as np
from gym.envs.mujoco import HumanoidEnv as HumanoidEnv
from gym import spaces

import random


# def mass_center(model, sim):
#     mass = np.expand_dims(model.body_mass, 1)
#     xpos = sim.data.xipos
#     return (np.sum(mass * xpos, 0) / np.sum(mass))


class HumanoidBodyEnv(HumanoidEnv):

    def __init__(self, max_episode_steps=200, eval_mode=False):
        self.task_dim = 3
        self._max_episode_steps = max_episode_steps
        self.action_scale = 1  # Mujoco environment initialization takes a step,
        super(HumanoidBodyEnv, self).__init__()

        # Override action space to make it range from  (-1, 1)
        assert (self.action_space.low == -self.action_space.high).all()
        self.action_scale = self.action_space.high[0]
        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=self.action_space.shape)  # Overriding original action_space which is (-0.4, 0.4, shape = (17, ))

        # save original cheetah properties (tasks are defined as ratios of these)
        self.original_mass = self.model.body_mass.copy()
        self.original_width = self.model.geom_size.copy()
        self.original_damp = self.model.dof_damping.copy()
        self.original_len = self.model.geom_size[2, :].copy()

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

    # def step(self, action):
    #     pos_before = np.copy(mass_center(self.model, self.sim)[:2])
    #
    #     rescaled_action = action * self.action_scale  # Scale the action from (-1, 1) to original.
    #     self.do_simulation(rescaled_action, self.frame_skip)
    #     pos_after = mass_center(self.model, self.sim)[:2]
    #
    #     alive_bonus = 5.0
    #     data = self.sim.data
    #     goal_direction = (np.cos(self._goal), np.sin(self._goal))
    #     lin_vel_cost = 0.25 * np.sum(goal_direction * (pos_after - pos_before)) / self.model.opt.timestep
    #     quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
    #     quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
    #     quad_impact_cost = min(quad_impact_cost, 10)
    #     reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
    #     qpos = self.sim.data.qpos
    #
    #     done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))
    #     # done = False
    #
    #     return self._get_obs(), reward, done, dict(reward_linvel=lin_vel_cost,
    #                                                reward_quadctrl=-quad_ctrl_cost,
    #                                                reward_alive=alive_bonus,
    #                                                reward_impact=-quad_impact_cost)

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

        self.model.geom_size[1, 0] = task[0] * self.original_width[1, 0]
        self.model.geom_size[3:, 0] = task[0] * self.original_width[3:, 0]
        for i in range(len(self.model.body_mass)):
            self.model.body_mass[i] = task[0] * self.original_mass[i]
        for i in range(len(self.model.dof_damping)):
            self.model.dof_damping[i] = task[1] * self.original_damp[i]
        self.model.geom_size[2, :] = task[2] * self.original_len

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
