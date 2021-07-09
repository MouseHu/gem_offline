import os
import time

import tensorflow as tf

from run.run_util import parse_args, create_action_noise, create_env, create_offline_env, save_args
from stable_baselines import logger
from stable_baselines.common.misc_util import set_global_seeds
from stable_baselines.td3 import GEMOffline


def run(env_id, seed, evaluation, gamma=0.99, **kwargs):
    # Create envs.
    env, dataset = create_offline_env(env_id)
    print(env.observation_space, env.action_space)
    if evaluation:
        eval_env, _ = create_offline_env(env_id)
    else:
        eval_env = None

    # Seed everything to make things reproducible.
    logger.info('seed={}, logdir={}'.format(seed, logger.get_dir()))
    tf.reset_default_graph()
    set_global_seeds(seed)
    env.seed(seed)
    if eval_env is not None:
        eval_env.seed(seed)

    # Disable logging for rank != 0 to avoid noise.
    start_time = time.time()

    policy = 'TD3LnMlpPolicy'
    model = GEMOffline(policy=policy, env=env, eval_env=eval_env, gamma=gamma, batch_size=128,
                       tau=0.005, policy_delay=2, learning_starts=25000,
                       action_noise=create_action_noise(env, "normal_0.1"),
                       buffer_size=len(env.get_dataset()['observations']), verbose=2,
                       n_cpu_tf_sess=10,
                       alpha=1, beta=-1, iterative_q=-1,
                       num_q=4, gradient_steps=200, max_step=kwargs['max_steps'], reward_scale=1., nb_eval_steps=10,
                       policy_kwargs={"layers": [400, 300]})

    print("model building finished")
    model.fill_dataset(dataset)
    print("fill dataset finished")
    model.learn(total_timesteps=kwargs['num_timesteps'])
    print("learning finished")
    env.close()
    if eval_env is not None:
        eval_env.close()

    logger.info('total runtime: {}s'.format(time.time() - start_time))


if __name__ == '__main__':
    args = parse_args()
    os.environ["OPENAI_LOGDIR"] = os.path.join(os.getenv("OPENAI_LOGDIR"), args["comment"])
    save_args(args)
    logger.configure()
    # Run actual script.
    run(**args)
