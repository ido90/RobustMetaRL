import random

import numpy as np

from gym.spaces import Box
from .half_cheetah import HalfCheetahEnv

MODEL_KEYS = ['actuator_gear', 'actuator_lengthrange', 'body_inertia', 'cam_ipd', 'geom_solmix',
              'geom_solref', 'jnt_margin', 'jnt_stiffness', 'light_pos', 'mat_rgba']
# MODEL_KEYS = ['cam_ipd', 'cam_mat0', 'dof_frictionloss', 'dof_solref', 'geom_friction',
#               'geom_gap', 'geom_rbound', 'geom_solimp', 'light_specular', 'mat_shininess']
# MODEL_KEYS = ['body_ipos', 'body_mass', 'body_pos', 'geom_gap', 'geom_margin',
#               'jnt_range', 'jnt_solref', 'jnt_stiffness', 'light_specular', 'mat_emission']

class HalfCheetahMultiEnv(HalfCheetahEnv):
    """Half-cheetah environment with varying body. The code is adapted from
    https://github.com/lmzintgraf/varibad/blob/master/environments/mujoco/half_cheetah_vel.py

    The half-cheetah follows the dynamics and rewards from MuJoCo.
    Its tasks correspond to different values of cheetah body mass, damping and torso length,
    sampled log-uniformly in the range of [50%, 200%] of their original values.
    """

    def __init__(self, max_episode_steps=200, eval_mode=False):
        self.eval_mode = eval_mode
        self._max_episode_steps = max_episode_steps
        self._time = 0
        self._return = 0
        self._last_return = 0
        self._curr_rets = []
        self.task = None
        super().__init__()
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(20,),
                                     dtype=np.float64)

        # save original cheetah properties (properties chosen randomly:)
        # sorted(np.random.choice([atr for atr in env.model.__dir__()
        #                          if type(getattr(env.model, atr))==np.ndarray and getattr(env.model, atr).dtype in (np.float32,np.float64)], 10, False))
        self.model_keys = MODEL_KEYS
        self.original_vecs = [getattr(self.model, k).copy()
                              for k in self.model_keys]
        self.task_dim = len(self.model_keys)

        self.set_task(self.sample_task())

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

        for i, k in enumerate(self.model_keys):
            for j in range(len(self.original_vecs[i])):
                getattr(self.model, k)[j] = task[i] * self.original_vecs[i][j]

        return task

    def get_task(self):
        return self.task

    # def seed(self, seed):
    #     random.seed(seed)
    #     np.random.seed(seed)

    def sample_task(self):
        return np.array([2 ** random.uniform(-0.5, 0.5)
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
