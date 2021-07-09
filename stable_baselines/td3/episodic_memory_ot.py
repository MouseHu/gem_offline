import numpy as np

from stable_baselines.td3.episodic_memory import EpisodicMemory, array_min2d


class EpisodicMemoryOT(EpisodicMemory):
    def __init__(self, buffer_size, state_dim, action_shape, obs_space, q_func, repr_func, obs_ph, action_ph, sess,
                 gamma=0.99, trajectory_length=5,
                 alpha=0.6, max_step=1000):
        super(EpisodicMemoryOT, self).__init__(buffer_size, state_dim, action_shape, obs_space, q_func, repr_func,
                                               obs_ph, action_ph, sess,
                                               gamma, alpha, max_step)
        # del self._q_values
        # self._q_values = -np.inf * np.ones((buffer_size + 1, 2))
        self.approximate_q_values = -np.inf * np.ones((buffer_size, 2))
        self.upper_bound = np.inf * np.ones(buffer_size)
        self.lower_bound = -np.inf * np.ones(buffer_size)
        self.batch_size = 16384
        self.max_step = max_step
        self.traj_length = trajectory_length

    def update_memory(self, q_base=0, use_knn=False, beta=-1):

        trajs = self.retrieve_trajectories()  # reversed!
        for traj in trajs:
            traj = list(reversed(traj))
            # print(np.array(traj))
            approximate_qs = self.compute_approximate_return(self.replay_buffer[traj], self.action_buffer[traj])
            approximate_qs = approximate_qs.reshape(-1)
            approximate_qs = np.append(approximate_qs,0)

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
                self.upper_bound[traj[i]] = min(backward_bound[i, :back_len])
                self.lower_bound[traj[i]] = max(forward_bound[i, :forward_len])

    def sample(self, batch_size, mix=False, priority=False):
        result = super().sample(batch_size, mix, priority)
        if result is None:
            return None
        idx = result['index0']
        result['upper_bound'] = array_min2d(self.upper_bound[idx])
        result['lower_bound'] = array_min2d(self.lower_bound[idx])
        return result
