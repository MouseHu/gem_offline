import matplotlib as mpl
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

mpl.use("TkAgg")
# file_dir = "/data1/hh/ddq_why/ddq_final/halfcheetah_ddq_alpha_0.5_with_tps_max_step_1001_0/episodic_memory.pkl"
# file_dir = "/data1/hh/130/ddq_supp/ant_ddq_max_step_100_max_step_100_beta_-1_4/episodic_memory.pkl"
file_dir = "/data1/hh/ddq_revisited/hopper_ddq_maxstep_1_3/episodic_memory.pkl"
# file_dir = "/data1/hh/ddq_revisited/hopper_ddq_update_200_lnmlp_beta_0.95_max_step_5_1/episodic_memory.pkl"
# file_dir = "C:/Users/Mouse Hu/Desktop/CEC/data/humanoid_ddq6_td_2/episodic_memory.pkl"
# file_dir = "C:/Users/Mouse Hu/Desktop/amc_data/_0.9_max_step_1_beta_-1_4/episodic_memory.pkl"
# file_dir = "C:/Users/Mouse Hu/Desktop/amc_data/_0.9_max_step_1000_beta_-1_0/episodic_memory.pkl"
# file_dir = "C:/Users/Mouse Hu/Desktop/CEC/data/episodic_memory.pkl"


with open(file_dir, "rb") as memory_file:
    memory = pkl.load(memory_file)
    print(memory.keys())

returns = memory["returns"]
q_values = memory["_q_values"]
states = memory["replay_buffer"]
done_buffer = memory["done_buffer"]
reward_buffer = memory["reward_buffer"]
pos = states[:, 1]
# vel = states[:,5:7]
end_points = np.where(done_buffer == True)[0].tolist()

trajs = [np.arange(x, y) for x, y in zip(end_points[:-1], end_points[1:])]
for i in range(5):
    n = np.random.randint(0, len(trajs))
    fig = plt.figure()
    traj = list(reversed(trajs[n]))
    # ax = Axes3D(fig)
    # print(len(traj),q_values[traj,:0].shape)
    plt.scatter(np.arange(len(traj)), q_values[traj,0] / 500, )
    plt.scatter(np.arange(len(traj)), returns[traj] / 500)
    plt.scatter(np.arange(len(traj)), pos[traj])
    # plt.scatter(np.arange(len(traj)), reward_buffer[traj]/10)
    # plt.scatter(np.arange(len(traj)),np.sqrt(vel[traj,0]**2))
    plt.legend(["q_values", "return", "pos_z","reward"])
    plt.show()
