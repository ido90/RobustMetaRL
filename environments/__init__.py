from gym.envs.registration import register

# Mujoco
# ----------------------------------------

# - randomised reward functions

register(
    'AntDir-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.ant_dir:AntDirEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

register(
    'AntDir2D-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.ant_dir:AntDir2DEnv',
            'max_episode_steps': 200},
    max_episode_steps=200,
)

register(
    'AntGoal-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.ant_goal:AntGoalEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

register(
    'AntMass-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.ant_mass:AntMassEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

register(
    'AntBody-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.ant_body:AntBodyEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

register(
    'AntVel-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.ant_vel:AntVelEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

register(
    'HalfCheetahDir-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.half_cheetah_dir:HalfCheetahDirEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

register(
    'HalfCheetahVel-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.half_cheetah_vel:HalfCheetahVelEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

register(
    'HalfCheetahMass-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.half_cheetah_mass:HalfCheetahMassEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

register(
    'HalfCheetahMulti-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.half_cheetah_multi:HalfCheetahMultiEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

register(
    'HalfCheetahMultiMass-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.half_cheetah_multimass:HalfCheetahMultiMassEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

register(
    'HalfCheetahBody-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.half_cheetah_body:HalfCheetahBodyEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

register(
    'HumanoidDir-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.humanoid_dir:HumanoidDirEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

register(
    'HumanoidVel-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.humanoid_vel:HumanoidVelEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

register(
    'HumanoidMass-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.humanoid_mass:HumanoidMassEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

register(
    'HumanoidBody-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.humanoid_body:HumanoidBodyEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

# - randomised dynamics

register(
    id='Walker2DRandParams-v0',
    entry_point='environments.mujoco.rand_param_envs.walker2d_rand_params:Walker2DRandParamsEnv',
    max_episode_steps=200
)

register(
    id='HopperRandParams-v0',
    entry_point='environments.mujoco.rand_param_envs.hopper_rand_params:HopperRandParamsEnv',
    max_episode_steps=200
)


# # 2D Navigation
# # ----------------------------------------
#
register(
    'PointEnv-v0',
    entry_point='environments.navigation.point_robot:PointEnv',
    kwargs={'goal_radius': 0.2,
            'max_episode_steps': 100,
            'goal_sampler': 'semi-circle'
            },
    max_episode_steps=100,
)

register(
    'SparsePointEnv-v0',
    entry_point='environments.navigation.point_robot:SparsePointEnv',
    kwargs={'goal_radius': 0.2,
            'max_episode_steps': 100,
            'goal_sampler': 'semi-circle'
            },
    max_episode_steps=100,
)

#
# # GridWorld
# # ----------------------------------------

register(
    'GridNavi-v0',
    entry_point='environments.navigation.gridworld:GridNavi',
    kwargs={'num_cells': 5, 'num_steps': 15},
)

register(
    'KhazadDum-v0',
    entry_point='environments.navigation.KhazadDum:KhazadDum',
    kwargs={},
    max_episode_steps=32,
)
