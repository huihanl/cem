import argparse
from collections import defaultdict
import heapq
from multiprocessing import Pool
from functools import partial
import os
import time

from plotting import plot_history

import roboverse
import numpy as np
import gym
from precog.predictive_env import PredictiveModel
import roboverse.bullet as bullet

unnorm_data_path = "/home/huihanl/bm-new/data/all_random_grasping_replayed/all.npy"

def normalize_by_dataset():
    if unnorm_data_path == "":
        return 0, 1
    unnorm_data = np.load(unnorm_data_path, allow_pickle=True)
    """
    for data_id in range(len(unnorm_data)):
        unnorm_data[data_id]["actions"] = unnorm_data[data_id]["actions"] + [np.array([0, 0, 0, 0])] * T
    """
    all_actions = []
    for traj_id in range(len(unnorm_data)):
        all_actions.extend(unnorm_data[traj_id]["actions"])
    all_actions = np.array(all_actions)
    mean = np.mean(all_actions, axis=0)
    stddev = np.std(all_actions, axis=0)
    if stddev[3] == 0.0:
        stddev[3] = 1
    return mean, stddev


mean, stddev = normalize_by_dataset()
print("mean: ", mean)
print("stddev: ", stddev)

class PredictiveModelEnvWrapper:

    def __init__(self, model_dir, num_execution_per_step, base_env=None, img_dim=48):
        self.predictive_model = PredictiveModel(model_dir)
        self.base_env = base_env
        self.img_dim = img_dim

        self.num_execution_per_step = num_execution_per_step
        self.past_length = self.predictive_model.past_length
        self.state_dim = self.predictive_model.state_dim  ## should be smaller than 11
        self.past = np.zeros([self.past_length, self.state_dim])
        self._set_action_space()
        self.observation_space = base_env.observation_space

    def step(self, action):
        z = action
        obs = self.base_env.get_observation()
        obs = obs["image"].reshape([self.img_dim, self.img_dim, 3]) * 255
        real_action = self.predictive_model.predict(self.past[-self.past_length:], obs, z)
        total_reward = 0
        for i in range(self.num_execution_per_step):
            first_predicted_action = real_action[0, 0, 0, i]
            x, y, z, theta = first_predicted_action[0], first_predicted_action[1], first_predicted_action[2], \
                             first_predicted_action[3]
            a = np.array([x, y, z, theta])
            a = (a * stddev) + mean
            obs, reward, done, info = self.base_env.step(a)
            total_reward += reward

        quat = obs["state"][3:7]
        theta = bullet.quat_to_deg(quat)[2]
        state = np.array([obs["state"][0], obs["state"][1], obs["state"][2], theta]).reshape([1, self.state_dim])

        self.past = np.concatenate([self.past, state], axis=0)
        return obs, total_reward, done, info

    def _set_action_space(self):
        act_dim = self.predictive_model.z_dim
        act_bound = 1
        act_high = np.ones(act_dim) * act_bound
        self.action_space = gym.spaces.Box(-act_high, act_high)

    def reset(self):
        self.past = np.zeros([self.past_length, self.state_dim])
        obs = self.base_env.reset()
        return obs

    def __getattr__(self, attr):
        return getattr(self.base_env, attr)

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)

    if not os.path.exists(directory):
        os.makedirs(directory)


def get_elite_indicies(num_elite, rewards):
    return heapq.nlargest(num_elite, range(len(rewards)), rewards.take)

def create_env():

    model_dir = "/home/huihanl/precog_nick/logs/esp_train_results/2020-07/" \
                "07-23-19-52-51_dataset.sawyer_dataset_no_append.SawyerDatasetNoAppend_bijection.basic_image_rnn.BasicImageRNNBijection"

    num_execution_per_step = 2
    single_obj_reward = 0
    trimodal_positions_choice = 0
    reward_type = "dense"
    all_random = True

    trimodal_positions = [(0.8187814400771692, 0.21049907010351596, -0.3415106684025205),
                          (0.739567302423451, 0.14341819851789023, -0.341380192135101),
                          (0.6694743281369817, 0.26676337932361205, -0.3440640126774397)]

    base_env = roboverse.make(
        "SawyerGraspOneV2-v0", gui=False, randomize=False,
        observation_mode="pixels_debug", reward_type=reward_type,
        single_obj_reward=single_obj_reward,
        normalize_and_flatten=True,
        all_random=all_random,
        trimodal_positions=trimodal_positions)

    img_width, img_height = base_env.obs_img_dim, base_env.obs_img_dim

    env = PredictiveModelEnvWrapper(model_dir, num_execution_per_step, base_env=base_env, img_dim=img_width)

    return env


def evaluate_z(z):
    env = create_env()
    env.reset()
    rewards = []
    for i in range(12):
        z_action = z[i*4: (i+1)*4]
        next_observation, reward, done, info = env.step(z_action)
        print("reward: ", reward)
        rewards.append(reward)
        if done:
            break
    return rewards[-1]


def run_cem(
        env_id,

        epochs=50,
        batch_size=4096,
        elite_frac=0.2,

        extra_std=2.0,
        extra_decay_time=10,

        num_process=8
):
    ensure_dir('./{}/'.format(env_id))

    start = time.time()
    num_episodes = epochs * num_process * batch_size
    print('expt of {} total episodes'.format(num_episodes))

    num_elite = int(batch_size * elite_frac)
    history = defaultdict(list)

    z_dim = 4 * 12
    means = np.zeros(z_dim)
    stds = np.ones(z_dim)

    for epoch in range(epochs):

        extra_cov = max(1.0 - epoch / extra_decay_time, 0) * extra_std**2

        zs = np.random.multivariate_normal(
            mean=means,
            cov=np.diag(np.array(stds**2) + extra_cov),
            size=batch_size
        )

        with Pool(num_process) as p:
            rewards = p.map(partial(evaluate_z), zs)

        rewards = np.array(rewards)

        indicies = get_elite_indicies(num_elite, rewards)
        elites = zs[indicies]

        means = elites.mean(axis=0)
        stds = elites.std(axis=0)

        history['epoch'].append(epoch)
        history['avg_rew'].append(np.mean(rewards))
        history['std_rew'].append(np.std(rewards))
        history['avg_elites'].append(np.mean(rewards[indicies]))
        history['std_elites'].append(np.std(rewards[indicies]))

        print(
            'epoch {} - {:2.1f} {:2.1f} pop - {:2.1f} {:2.1f} elites'.format(
                epoch,
                history['avg_rew'][-1],
                history['std_rew'][-1],
                history['avg_elites'][-1],
                history['std_elites'][-1]
            )
        )

    end = time.time()
    expt_time = end - start
    print('expt took {:2.1f} seconds'.format(expt_time))

    plot_history(history, env_id, num_episodes, expt_time)
    num_optimal = 3
    print('epochs done - evaluating {} best zs'.format(num_optimal))

    best_z_rewards = [evaluate_z(z) for z in elites[:num_optimal]]
    print('best rewards - {} across {} samples'.format(best_z_rewards, num_optimal))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default="SawyerGraspOneV2-v0")
    parser.add_argument('--num_process', default=16, nargs='?', type=int)
    parser.add_argument('--epochs', default=50, nargs='?', type=int)
    parser.add_argument('--batch_size', default=4096, nargs='?', type=int)
    args = parser.parse_args()
    print(args)

    run_cem(args.env, num_process=args.num_process, epochs=args.epochs, batch_size=args.batch_size)
