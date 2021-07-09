from stable_baselines.common import tf_util, SetVerbosity
from stable_baselines.td3.td3_mem_ddq import TD3MemDDQ
from d4rl import sequence_dataset
from stable_baselines import logger
import time
from stable_baselines.common.schedules import get_schedule_fn
import numpy as np
from stable_baselines.common.math_util import safe_mean, unscale_action, scale_action, reward2return
import os
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.common import tf_util, OffPolicyRLModel, SetVerbosity, TensorboardWriter
from collections import deque


class GEMOffline(TD3MemDDQ):
    def __init__(self, policy, env, eval_env=None, gamma=0.99, learning_rate=3e-4,
                 buffer_size=50000,
                 learning_starts=100, train_freq=100, gradient_steps=100, batch_size=128,
                 tau=0.005, policy_delay=1, qvalue_delay=1, action_noise=None, max_step=1000,
                 nb_eval_steps=1000, alpha=0.5, beta=-1, num_q=4, iterative_q=True, reward_scale=1.,
                 target_policy_noise=0.2, target_noise_clip=0.5, start_policy_learning=0,
                 random_exploration=0.0, verbose=0, tensorboard_log=None,
                 _init_setup_model=True, policy_kwargs=None, double_type="identical",
                 full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None):
        super(GEMOffline, self).__init__(policy, env, eval_env, gamma, learning_rate,
                                         buffer_size,
                                         learning_starts, train_freq, gradient_steps, batch_size,
                                         tau, policy_delay, qvalue_delay, action_noise,max_step,
                                         nb_eval_steps, alpha, beta, num_q, iterative_q, reward_scale,
                                         target_policy_noise, target_noise_clip, start_policy_learning,
                                         random_exploration, verbose, tensorboard_log,
                                         _init_setup_model, policy_kwargs,double_type,
                                         full_tensorboard_log, seed, n_cpu_tf_sess)

    def fill_dataset(self, dataset: sequence_dataset):
        for seq in dataset:
            observations, actions, dones, rewards, truly_dones = seq['observations'], seq['actions'], seq[
                'timeouts'], seq['rewards'], seq['terminals']
            trajectory = [(obs, action, 0, 0, reward, truly_done, done) for
                          obs, action, reward, truly_done, done in
                          zip(observations, actions, rewards, dones, truly_dones)]
            self.memory.update_sequence_with_qs(trajectory)

    def learn(self, total_timesteps, eval_interval=10000, update_interval=10000, callback=None,
              log_interval=4, tb_log_name="TD3", reset_num_timesteps=True, replay_wrapper=None):

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)
        callback = self._init_callback(callback)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:
            self._setup_learn()

            # Transform to callable if needed
            self.learning_rate = get_schedule_fn(self.learning_rate)
            # Initial learning rate
            current_lr = self.learning_rate(1)

            start_time = time.time()
            train_time, update_time = 0, 0

            callback.on_training_start(locals(), globals())
            callback.on_rollout_start()

            for step in range(total_timesteps + 1):
                # Update policy, critics and target networks
                if self.num_timesteps % self.train_freq == 0:
                    cur_time = time.time()
                    callback.on_rollout_end()
                    self.sess.run(self.target_ops)
                    self.memory.update_memory(self.q_base, beta=self.beta)
                    update_time += time.time() - cur_time
                    cur_time = time.time()
                    mb_infos_vals = []
                    for grad_step in range(self.gradient_steps):
                        # Compute current learning_rate
                        frac = 1.0 - step / total_timesteps
                        current_lr = self.learning_rate(frac)
                        if_train_policy = ((step + grad_step) % self.policy_delay == 0)
                        if_train_q = (step + grad_step) % self.qvalue_delay == 0
                        mb_infos_vals.append(
                            self._train_step(step, writer, current_lr, if_train_policy, if_train_q))

                        # Log losses and entropy, useful for monitor training
                        if len(mb_infos_vals) > 0:
                            infos_values = np.array(np.mean(mb_infos_vals, axis=0)).reshape(-1)

                        # update memory much more frequenctly
                        # print("updating memory")
                        train_time += time.time() - cur_time
                        callback.on_rollout_start()

                if step % eval_interval == 0:
                    # Evaluate.
                    self.evaluate(self.nb_eval_steps)
                self.num_timesteps += 1
                if step % 200000 == 0:  # save every 200k step
                    self.save(os.path.join(os.getenv("OPENAI_LOGDIR"), "model.pkl"))
                # Display training infos
                if self.verbose >= 1 and log_interval is not None and step % log_interval == 0:
                    fps = int(step / (time.time() - start_time))
                    logger.logkv("current_lr", current_lr)
                    logger.logkv("fps", fps)
                    logger.logkv('time_elapsed', int(time.time() - start_time))
                    logger.logkv('train_time', int(train_time))
                    logger.logkv('update_time', int(update_time))
                    if len(infos_values) > 0:
                        for (name, val) in zip(self.infos_names, infos_values):
                            logger.logkv(name, val)
                    logger.logkv("total timesteps", self.num_timesteps)
                    logger.dumpkvs()
                    infos_values = []

            callback.on_training_end()
            return self
