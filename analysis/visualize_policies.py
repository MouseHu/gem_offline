import os
import numpy as np
import time
from run.run_util import create_env, create_offline_env
from stable_baselines.td3.gem_offline import GEMOffline
from stable_baselines.td3.td3_mem_ddq import TD3MemDDQ
import matplotlib.pyplot as plt

# model_dir = "/data1/hh/130/ddq_debug/ant_ddq_no_truly_done_max_step_1_beta_-1_4/model.pkl"


# model_dir = "/data1/hh/130/ddq_supp/ant_ddq_max_step_1001_max_step_1001_beta_-1_3/model.pkl"
model_dir = "/home/hh/log_gem_offline/gem/ant_umaze_gem_test_0/model.pkl"


def load_agent(load_dir, agent_func, env):
    agent = agent_func.load(load_dir, env)
    # agent.setup_model()
    return agent


# env = create_env("Ant-v2", 0, 0)
env, _ = create_offline_env("antmaze-umaze-diverse-v0")
agent = load_agent(model_dir, GEMOffline, env)
num_episode = 3

rewardss = []
for ep in range(num_episode):
    print("episode: {}".format(ep))
    obs = env.reset()
    done = False
    rewards = []
    while not done:
        action = agent.policy_tf.step(obs[None]).flatten()
        new_obs, reward, done, info = env.step(action)
        # truly_done = info.get("truly_done",done)
        # done = truly_done
        rewards.append(reward)
        obs = new_obs
        env.render()
        # print(done)
        time.sleep(0.01)
    rewardss.append(rewards)

for rewards in rewardss:
    discount_return = 0
    for i, r in enumerate(rewards):
        discount_return += 0.99 ** i * r
    print(discount_return)
    plt.plot(np.arange(len(rewards)), np.array(rewards))

plt.show()
