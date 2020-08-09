import argparse
from collections import defaultdict
import heapq
from multiprocessing import Pool
from functools import partial
import os
import time

from tqdm import tqdm
from plotting import plot_history

import roboverse
import numpy as np
import gym
from precog.predictive_env import PredictiveModel
import roboverse.bullet as bullet
from datetime import datetime
unnorm_data_path_server = "/nfs/kun1/users/huihanl/all.npy"
unnorm_data_path_local = "/home/huihanl/bm-new/data/all_random_grasping_replayed/all.npy"

def normalize_by_dataset():

    mean = np.array([0.07205797, -0.02857389, -0.28423747, -0.64467074])
    stddev = np.array([0.14842513, 0.14714153, 0.10333442, 1.15283709])
    return mean, stddev


mean, stddev = normalize_by_dataset()

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


def get_elite_indicies_num(num_elite, returns):
    return heapq.nlargest(num_elite, range(len(returns)), returns.take)


def get_elite_indicies_success(successes):
    num_elite = np.count_nonzero(successes)
    return heapq.nlargest(num_elite, range(len(successes)), successes.take)


def get_elite_indicies(num_elite, returns, successes, only_success_elite):
    if only_success_elite:
        indexes = get_elite_indicies_success(successes)
        if len(indexes) < 3:
            indexes.extend(get_elite_indicies_num(3, returns))
    else:
        indexes = get_elite_indicies_num(num_elite, returns)
    return indexes


def create_env(randomize, reward_type, env_index, single_obj_reward):

    model_dir_aws = "/home/ubuntu/07-23-19-52-51_dataset.sawyer_dataset_no_append.SawyerDatasetNoAppend_bijection.basic_image_rnn.BasicImageRNNBijection"
    num_execution_per_step = 2

    trimodal_positions6 = [(0.8187814400771692, 0.21049907010351596, -0.3415106684025205),
                          (0.739567302423451, 0.14341819851789023, -0.341380192135101),
                          (0.6694743281369817, 0.26676337932361205, -0.3440640126774397)]

    trimodal_positions5 = [(0.6838513781345804, 0.13232607273283675, -0.34202212957838085), 
                          (0.765710933262148, 0.17467922332524338, -0.3351583030520283), 
                          (0.6789658820114347, 0.2548283056922718, -0.34103265910465325)]

    trimodal_positions0 = [(0.7044727231056753, 0.2885882337328971, -0.3434554430497813), 
                           (0.6950925950841189, 0.10769150255843486, -0.3408203519762647), 
                           (0.8303787805809874, 0.12114265344994643, -0.34381720097869056)]

    trimodal_positions1 = [(0.7543401984506464, 0.21230734165805112, -0.3400822998707757),
                          (0.7250127303935135, 0.1172465013108163, -0.34050488361050935),
                          (0.668601621056849, 0.20450849328681078, -0.34356676661067254)]
                        
    trimodal_positions2 = [(0.6623195604074853, 0.1420046836817064, -0.34036979015824237),
                          (0.7490314063602679, 0.11150642565578632, -0.34010550517840776),
                          (0.8313050761244212, 0.186704691256355, -0.3444288731959656)]
                        
    trimodal_positions3 = [(0.702806187582968, 0.282862951083425, -0.34329308543453),
                          (0.8053413878164436, 0.15063075895870554, -0.34222136237585643),
                          (0.6598757532001869, 0.10674964260605753, -0.34047814967568574)]
                        
    trimodal_positions4 = [(0.7935222824622936, 0.19097678049219627, -0.336295087280328),
                           (0.6742503469035555, 0.1988108314637719, -0.3439745727367933),
                           (0.6661682273867254, 0.09909463325237348, -0.3399316482536911)]

    trimodal_positions_choice = [trimodal_positions0, trimodal_positions1, trimodal_positions2, 
                                 trimodal_positions3, trimodal_positions4, trimodal_positions5, trimodal_positions6]

    trimodal_positions = trimodal_positions_choice[env_index]

    base_env = roboverse.make(
        "SawyerGraspOneV2-v0", gui=False, randomize=randomize,
        observation_mode="pixels_debug", reward_type=reward_type,
        single_obj_reward=single_obj_reward,
        normalize_and_flatten=True,
        all_random=True,
        trimodal_positions=trimodal_positions)

    img_width, img_height = base_env.obs_img_dim, base_env.obs_img_dim

    env = PredictiveModelEnvWrapper(model_dir_aws, num_execution_per_step, base_env=base_env, img_dim=img_width)

    return env

def evaluate_z(z, randomize, reward_type, output_dir, z_id, env_index, single_obj_reward):
    env = create_env(randomize, reward_type, env_index, single_obj_reward)

    returns_list = []
    success_list = []
    for trial in range(10):
        env.reset()
        rewards = []
        success = 0
        images = []
        for i in range(12):
            z_action = z[i*4: (i+1)*4]
            next_observation, reward, done, info = env.step(z_action)
            rewards.append(reward)
            images.append(env.render_obs())
            if done:
                if info["grasp_success"]:
                    success = 1
                break
        returns = sum(rewards)
        returns_list.append(returns)
        success_list.append(success)
    print("at z id: ", z_id)
    print("returns list: ", returns_list)
    print("success list: ", success_list)
    mean_returns = mean(returns_list)
    mean_success = mean(success_list)
    print("mean returns: ", mean_returns)
    print("mean success: ", mean_success)
    return (mean_returns, mean_success)



def run_cem(
        env_id,

        epochs=50,
        batch_size=4096,
        elite_frac=0.0625,

        randomize=True,
        only_success_elite=False,
        reward_type="sparse",
        env_index=0,
        single_obj_reward=-1,
        z_dir=None,

        extra_std=2.0,
        extra_decay_time=10,

        num_process=8
):

    now = datetime.now()
    time_now = now.strftime("%m-%d-%H-%M-%S")
    output_dir = './eval_z_repeat/{}/'.format(time_now)
    print("output_dir: ", output_dir)
    ensure_dir(output_dir)

    num_optimal = 2

    elites = np.load(z_dir, allow_pickle=True)

    returns_successes = []
    for n in range(num_optimal):
        z = elites[n]
        returns_successes_n = evaluate_z(z, randomize=randomize, reward_type=reward_type, output_dir=output_dir, 
                                         z_id=n, env_index=env_index, single_obj_reward=single_obj_reward)
        returns_successes.append(returns_successes_n)

    returns_mean = [rs[0] for rs in returns_successes]
    successes_mean = [rs[1] for rs in returns_successes]

    returns_mean = np.array(returns_mean)
    successes_mean = np.array(successes_mean)
    print("returns mean: ", returns_mean)
    print("successes mean: ", successes_mean)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default="SawyerGraspOneV2-v0")
    parser.add_argument('--num_process', default=8, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=4096, type=int)
    parser.add_argument('--randomize', default=False, type=bool)
    parser.add_argument('--only_success_elite', default=False, type=bool)
    parser.add_argument('--reward_type', default="sparse", type=str)
    parser.add_argument('--env_index', default=0, type=int)
    parser.add_argument('--single_obj_reward', required=True, type=int)
    parser.add_argument('--z_dir', required=True, type=str)
    args = parser.parse_args()
    print(args)

    run_cem(args.env,
            num_process=args.num_process,
            epochs=args.epochs,
            randomize=args.randomize,
            batch_size=args.batch_size,
            only_success_elite=args.only_success_elite,
            reward_type=args.reward_type,
            env_index=args.env_index,
            single_obj_reward=args.single_obj_reward,
            z_dir=args.z_dir,
            )
