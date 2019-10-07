import gym
import sys

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import A2C

sys.path.append(r"C:\Users\...\gym-T77")

# multiprocess environment
n_cpu = 1
env = SubprocVecEnv([lambda: gym.make('gym_t77:t77-v0') for i in range(n_cpu)])

model = A2C(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=25000)
model.save("a2c_t77")

del model  # remove to demonstrate saving and loading

model = A2C.load("a2c_t77")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()