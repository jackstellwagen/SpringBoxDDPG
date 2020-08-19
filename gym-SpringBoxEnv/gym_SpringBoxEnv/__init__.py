from gym.envs.registration import register

register(
    id='SpringBoxEnv-v0',
    entry_point='gym_SpringBoxEnv.envs:SpringBoxEnv',
)
