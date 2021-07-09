import gym
import d4rl  # Import required to register environments

# Create the environment
env = gym.make('antmaze-umaze-diverse-v0')

# d4rl abides by the OpenAI gym interface
env.reset()
env.step(env.action_space.sample())

# Each task is associated with a dataset
# dataset contains observations, actions, rewards, terminals, and infos
dataset = env.get_dataset()
print(dataset['observations'])  # An N x dim_observation Numpy array of observations
print(len(dataset['observations']))
# Alternatively, use d4rl.qlearning_dataset which
# also adds next_observations.
# dataset = d4rl.qlearning_dataset(env)
dataset = d4rl.sequence_qlearning_dataset(env)

i = 0
for d in dataset:
    if i == 0:
        print(d.keys())
        print(type(d))
    print(len(d['actions']))
    print((d['rewards']>0).any(),(d['rewards']<0).any())
    i += 1

print("total trajs:", i)
