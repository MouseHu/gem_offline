import numpy as np


class TestOT(object):
    def __init__(self):
        self.gamma = 0.99
        self.traj_length = 5
        self.qs = 0.1 * np.arange(10)
        self.reward_buffer = np.arange(10)
        self.upper_bound = np.zeros(10)
        self.lower_bound = np.zeros(10)

    def retrieve_trajectories(self):
        return [[i for i in range(10)]]

    def update_memory(self):
        trajs = self.retrieve_trajectories()  # reversed!
        for traj in trajs:
            # traj = list(reversed(traj))
            approximate_qs = self.qs[traj]
            approximate_qs = approximate_qs.reshape(-1)
            approximate_qs = np.append(approximate_qs, 0)

            forward_return = np.zeros((len(traj), self.traj_length))  # R_t:R_t+\tau
            forward_bound = np.zeros((len(traj), self.traj_length))  # R_t:R_t+\tau
            backward_return = np.zeros((len(traj), self.traj_length))  # R_t-\tau:R_t
            backward_bound = np.zeros((len(traj), self.traj_length))  # R_t-\tau:R_t

            forward_return[:, 0] = self.reward_buffer[traj]
            for l in range(1, self.traj_length):
                forward_return[:len(traj) - l, l] = forward_return[:len(traj) - l, l - 1] + self.gamma ** l * \
                                                    self.reward_buffer[traj[l:]]

            for l in range(self.traj_length):
                backward_return[l:, l] = forward_return[:len(traj) - l, l]

            for l in range(self.traj_length):
                forward_bound[:len(traj) - l, l] = forward_return[:len(traj) - l, l] + self.gamma ** (
                        l + 1) * approximate_qs[l + 1:]

            backward_bound[:, 0] = approximate_qs[:-1]
            for l in range(self.traj_length - 1):
                backward_bound[(l + 1):, l + 1] = self.gamma ** (-l - 1) * (
                        approximate_qs[:len(traj) - l - 1] - backward_return[l:-1, l])

            for i in range(len(traj)):
                back_len = min(self.traj_length, i + 1)
                forward_len = min(self.traj_length, len(traj) - i)
                self.upper_bound[i] = min(backward_bound[i, :back_len])
                self.lower_bound[i] = max(forward_bound[i, :forward_len])


t = TestOT()
t.update_memory()
print(t.upper_bound)
print(t.lower_bound)
