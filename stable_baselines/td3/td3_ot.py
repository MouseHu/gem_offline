import tensorflow as tf

from stable_baselines.common import tf_util, SetVerbosity
from stable_baselines.common.buffers import ReplayBuffer
from stable_baselines.common.math_util import unscale_action
from stable_baselines.td3.episodic_memory_ot import EpisodicMemoryOT
from stable_baselines.td3.td3_mem_many import TD3MemUpdateMany


class TD3MemOT(TD3MemUpdateMany):
    """
    Twin Delayed DDPG (TD3)
    Addressing Function Approximation Error in Actor-Critic Methods.

    Original implementation: https://github.com/sfujim/TD3
    Paper: https://arxiv.org/pdf/1802.09477.pdf
    Introduction to TD3: https://spinningup.openai.com/en/latest/algorithms/td3.html

    :param policy: (TD3Policy or str) The policy model to use (MlpPolicy, CnnPolicy, LnMlpPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) the discount factor
    :param learning_rate: (float or callable) learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values and Actor networks)
        it can be a function of the current progress (from 1 to 0)
    :param buffer_size: (int) size of the replay buffer
    :param batch_size: (int) Minibatch size for each gradient update
    :param tau: (float) the soft update coefficient ("polyak update" of the target networks, between 0 and 1)
    :param policy_delay: (int) Policy and target networks will only be updated once every policy_delay steps
        per training steps. The Q values will be updated policy_delay more often (update every training step).
    :param action_noise: (ActionNoise) the action noise type. Cf DDPG for the different action noise type.
    :param target_policy_noise: (float) Standard deviation of Gaussian noise added to target policy
        (smoothing noise)
    :param target_noise_clip: (float) Limit for absolute value of target policy smoothing noise.
    :param train_freq: (int) Update the model every `train_freq` steps.
    :param learning_starts: (int) how many steps of the model to collect transitions for before learning starts
    :param gradient_steps: (int) How many gradient update after each step
    :param random_exploration: (float) Probability of taking a random action (as in an epsilon-greedy strategy)
        This is not needed for TD3 normally but can help exploring when using HER + TD3.
        This hack was present in the original OpenAI Baselines repo (DDPG + HER)
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        Note: this has no effect on TD3 logging for now
    :param seed: (int) Seed for the pseudo-random generators (python, numpy, tensorflow).
        If None (default), use random seed. Note that if you want completely deterministic
        results, you must set `n_cpu_tf_sess` to 1.
    :param n_cpu_tf_sess: (int) The number of threads for TensorFlow operations
        If None, the number of cpu of the current machine will be used.
    """

    def __init__(self, policy, env, eval_env, gamma=0.99, learning_rate=3e-4,
                 buffer_size=50000,
                 learning_starts=100, train_freq=100, gradient_steps=100, batch_size=128,
                 tau=0.005, policy_delay=2, qvalue_delay=1, max_step=1000, action_noise=None,
                 nb_eval_steps=5, alpha=0.5, beta=-1, num_q=4, iterative_q=True, reward_scale=1.,
                 target_policy_noise=0.2, target_noise_clip=0.5, start_policy_learning=0, bound_ratio=0.8,
                 random_exploration=0.0, verbose=0, tensorboard_log=None,
                 _init_setup_model=True, policy_kwargs=None, traj_length=5,
                 full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None):

        print("TD3 Optimality Tightening Agent here")
        self.bound_ratio = bound_ratio
        self.trajectory_length = traj_length
        self.final_target = None
        super(TD3MemOT, self).__init__(policy, env, eval_env, gamma, learning_rate,
                                       buffer_size,
                                       learning_starts, train_freq, gradient_steps, batch_size,
                                       tau, policy_delay, qvalue_delay, max_step, action_noise,
                                       nb_eval_steps, alpha, beta, num_q, iterative_q, reward_scale,
                                       target_policy_noise, target_noise_clip, start_policy_learning,
                                       random_exploration, verbose, tensorboard_log,
                                       _init_setup_model, policy_kwargs,
                                       full_tensorboard_log, seed, n_cpu_tf_sess)

    def _get_pretrain_placeholders(self):
        policy = self.policy_tf
        # Rescale
        policy_out = unscale_action(self.action_space, self.policy_out)
        return policy.obs_ph, self.actions_ph, policy_out

    def setup_model(self):
        # print("setup model ",self.observation_space.shape)
        with SetVerbosity(self.verbose):
            self.graph = tf.Graph()
            with self.graph.as_default():
                self.set_random_seed(self.seed)
                self.sess = tf_util.make_session(num_cpu=self.n_cpu_tf_sess, graph=self.graph)

                self.replay_buffer = ReplayBuffer(self.buffer_size)

                with tf.variable_scope("input", reuse=False):
                    # Create policy and target TF objects
                    self.policy_tf = self.policy(self.sess, self.observation_space, self.action_space,
                                                 **self.policy_kwargs)
                    self.target_policy_tf = self.policy(self.sess, self.observation_space, self.action_space,
                                                        **self.policy_kwargs)

                    # Initialize Placeholders
                    self.observations_ph = self.policy_tf.obs_ph
                    # Normalized observation for pixels
                    self.processed_obs_ph = self.policy_tf.processed_obs
                    self.next_observations_ph = self.target_policy_tf.obs_ph
                    self.processed_next_obs_ph = self.target_policy_tf.processed_obs
                    self.action_target = self.target_policy_tf.action_ph
                    self.terminals_ph = tf.placeholder(tf.float32, shape=(None, 1), name='terminals')
                    self.rewards_ph = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
                    self.actions_ph = tf.placeholder(tf.float32, shape=(None,) + self.action_space.shape,
                                                     name='actions')
                    self.upper_bound_ph = tf.placeholder(tf.float32, shape=(None, 1),
                                                         name='upperbound')
                    self.lower_bound_ph = tf.placeholder(tf.float32, shape=(None, 1),
                                                         name='lowerbound')
                    self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")

                with tf.variable_scope("model", reuse=False):
                    # Create the policy
                    self.policy_out = policy_out = self.policy_tf.make_actor(self.processed_obs_ph)
                    # Use two Q-functions to improve performance by reducing overestimation bias
                    qfs = self.policy_tf.make_many_critics(self.processed_obs_ph, self.actions_ph,
                                                           scope="buffer_values_fn")
                    # Q value when following the current policy
                    self.qfs = qfs
                    self.qfs_pi = self.policy_tf.make_many_critics(self.processed_obs_ph,
                                                                   policy_out, scope="buffer_values_fn", reuse=True)

                with tf.variable_scope("target", reuse=False):
                    # Create target networks
                    target_policy_out = self.target_policy_tf.make_actor(self.processed_next_obs_ph)
                    # Q values when following the target policy
                    qfs_target = self.target_policy_tf.make_many_critics(self.processed_next_obs_ph,
                                                                         target_policy_out,
                                                                         scope="buffer_values_fn",
                                                                         reuse=False)

                    self.qfs_target = qfs_target
                    self.qfs_target_no_pi = self.target_policy_tf.make_many_critics(
                        self.processed_obs_ph,
                        self.actions_ph,
                        scope="buffer_values_fn", reuse=True)

                with tf.variable_scope("loss", reuse=False):
                    # Take the min of the two target Q-Values (clipped Double-Q Learning)
                    min_qf_target = tf.reduce_mean(qfs_target, axis=0) - self.q_base
                    # min_qf_target = tf.minimum(qf1_target, qf2_target)
                    print("here", min_qf_target.shape)
                    # Targets for Q value regression
                    q_backup = tf.stop_gradient(
                        self.rewards_ph +
                        (1 - self.terminals_ph) * self.gamma * min_qf_target
                    )
                    self.q_backup = q_backup
                    # Compute Q-Function loss

                    # Method 2
                    self.final_target = (1 - self.bound_ratio) * q_backup + 1. / 2 * self.bound_ratio * (
                            self.upper_bound_ph + self.lower_bound_ph)
                    alpha = .5
                    qfs_loss = tf.reduce_mean(
                        tf.nn.leaky_relu(-self.final_target + qfs - self.q_base, alpha=alpha) ** 2)
                    # qfs_loss = tf.reduce_mean((qfs - self.qvalues_ph) ** 2)
                    target_buffer_diff = tf.reduce_mean(((
                                                                 self.upper_bound_ph + self.lower_bound_ph) / 2 - q_backup) ** 2)
                    self.target_buffer_diff = target_buffer_diff
                    qvalues_losses = qfs_loss

                    self.policy_loss = policy_loss = -tf.reduce_mean(self.qfs_pi)

                    # Policy loss: maxsimise q value

                    # Policy train op
                    # will be called only every n training steps,
                    # where n is the policy delay
                    policy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
                    policy_train_op = policy_optimizer.minimize(policy_loss,
                                                                var_list=tf_util.get_trainable_vars('model/pi'))
                    self.policy_train_op = policy_train_op

                    # Q Values optimizer
                    qvalues_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
                    qvalues_params = tf_util.get_trainable_vars('model/values_fn/') + tf_util.get_trainable_vars(
                        'model/buffer_values_fn/')

                    # Q Values and policy target params
                    source_params = tf_util.get_trainable_vars("model/")
                    target_params = tf_util.get_trainable_vars("target/")

                    # Polyak averaging for target variables
                    # self.target_ops = [
                    #     tf.assign(target, (1 - self.tau) * target + self.tau * source)
                    #     for target, source in zip(target_params, source_params)
                    # ]
                    self.target_ops = [
                        tf.assign(target,
                                  (1 - self.tau) ** self.gradient_steps * target +
                                  (1 - (1 - self.tau) ** self.gradient_steps) * source)
                        for target, source in zip(target_params, source_params)
                    ]
                    # self.target_ops = [
                    #     tf.assign(target, source)
                    #     for target, source in zip(target_params, source_params)
                    # ]
                    # Initializing target to match source variables
                    target_init_op = [
                        tf.assign(target, source)
                        for target, source in zip(target_params, source_params)
                    ]

                    train_values_op = qvalues_optimizer.minimize(qvalues_losses, var_list=qvalues_params)

                    self.infos_names = ['qfs_loss', 'target_buffer_diff', "upper_bound", "lower_bound", "final_target"]
                    # All ops to call during one training step
                    self.step_ops = [qfs_loss, target_buffer_diff, tf.reduce_mean(self.upper_bound_ph),
                                     tf.reduce_mean(self.lower_bound_ph), tf.reduce_mean(self.final_target),
                                     qfs, train_values_op]

                    # Monitor losses and entropy in tensorboard
                    tf.summary.scalar('policy_loss', policy_loss)
                    tf.summary.scalar('qfs_loss', qfs_loss)
                    tf.summary.scalar('target_buffer_diff', target_buffer_diff)
                    tf.summary.scalar('upper_bound', tf.reduce_mean(self.upper_bound_ph))
                    tf.summary.scalar('lower_bound', tf.reduce_mean(self.lower_bound_ph))
                    tf.summary.scalar('final_target', tf.reduce_mean(self.final_target))
                    # tf.summary.scalar('qf4_loss', qf4_loss)
                    tf.summary.scalar('learning_rate', tf.reduce_mean(self.learning_rate_ph))

                # Retrieve parameters that must be saved
                self.params = tf_util.get_trainable_vars("model")
                self.target_params = tf_util.get_trainable_vars("target/")

                # Initialize Variables and target network
                with self.sess.as_default():
                    self.sess.run(tf.global_variables_initializer())
                    self.sess.run(target_init_op)

                self.summary = tf.summary.merge_all()

                self.memory = EpisodicMemoryOT(self.buffer_size, state_dim=1,
                                               obs_space=self.observation_space,
                                               action_shape=self.action_space.shape,
                                               q_func=self.qfs_target, repr_func=None,
                                               obs_ph=self.processed_next_obs_ph,
                                               action_ph=self.actions_ph, sess=self.sess, gamma=self.gamma,
                                               trajectory_length=self.trajectory_length,
                                               max_step=self.max_step)

    def _train_step(self, step, writer, learning_rate, update_policy, update_q):
        # Sample a batch from the replay buffer
        # batch = self.replay_buffer.sample(self.batch_size, env=self._vec_normalize_env)
        # cur_time = time.time()
        batch = self.memory.sample(self.batch_size, mix=False)
        if batch is None:
            return 0, 0
        batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones, batch_returns = batch['obs0'], batch[
            'actions'], batch['rewards'], batch['obs1'], batch['terminals1'], batch['return']
        batch_upper_bound = batch['upper_bound']
        batch_lower_bound = batch['lower_bound']
        # print(batch_lower_bound)
        # print(batch_upper_bound)
        # print("Sample time: ",time.time()-cur_time)
        # cur_time = time.time()
        # ot_target = (batch_upper_bound + batch_lower_bound) / 2
        feed_dict = {
            self.observations_ph: batch_obs,
            self.actions_ph: batch_actions,
            self.next_observations_ph: batch_next_obs,
            self.rewards_ph: batch_rewards.reshape(self.batch_size, -1),
            self.terminals_ph: batch_dones.reshape(self.batch_size, -1),
            self.learning_rate_ph: learning_rate,
            self.upper_bound_ph: batch_upper_bound.reshape(self.batch_size, -1),
            self.lower_bound_ph: batch_lower_bound.reshape(self.batch_size, -1)
        }
        # print("training ",batch_obs.shape)
        if update_q:
            step_ops = self.step_ops
        else:
            step_ops = []
        if update_policy:
            # Update policy and target networks
            step_ops = step_ops + [self.policy_train_op, self.policy_loss]
        if not step_ops:
            return 0, 0  # not updating q nor policy
        # Do one gradient step
        # and optionally compute log for tensorboard
        if writer is not None:
            out = self.sess.run([self.summary] + step_ops, feed_dict)
            summary = out.pop(0)
            writer.add_summary(summary, step)
        else:
            out = self.sess.run(step_ops, feed_dict)

        # Unpack to monitor losses
        qfs_loss, target_buffer_diff, upper, lower, final_target, *_values = out
        # print("Network time: ",time.time()-cur_time)
        return qfs_loss, target_buffer_diff, upper, lower, final_target
