# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The environment class for MonoBeast."""
from collections.abc import Iterable

import numpy as np
import torch
import gym
import timeit

from utils.dotdict import DotDict
from wordcraft.env import WordCraftEnv


def _format_env_output(output):
    return output.view((1, 1) + output.shape)  # (...) -> (T,B,...).

def env_state_has_instr(state):
    if 'instr' in state:
        if 'instr_length' in state:
            return True
        else:
            raise ValueError('Gym env state has instr but no instr_length attribute.')
    return False

# Create WordCraft env from flags here
def create_gym_env(flags, seed=None):
    env_name_lower = flags.env.lower()
    if 'wordcraft' in env_name_lower:
        max_depth = max(flags.depths)
        env = gym.make(
            flags.env, 
            seed=seed,
            data_path=flags.data_path,
            recipe_book_path=flags.recipe_book_path,
            max_depth=max_depth, 
            split=flags.split, 
            train_ratio=flags.train_ratio, 
            feature_type=flags.feature_type,
            random_feature_size=flags.random_feature_size, 
            shuffle_features=flags.shuffle_features,
            num_distractors=flags.num_distractors, 
            uniform_distractors=flags.uniform_distractors,
            max_mix_steps=flags.max_mix_steps,
            subgoal_rewards=flags.subgoal_rewards,
        )
        return WordCraft2Dict(flags, env)
    else:
        raise ValueError(f'Environment {flags.env} is not supported.')


class WordCraft2Dict(gym.ObservationWrapper):
    def __init__(self, flags, env):
        gym.ObservationWrapper.__init__(self, env)

        # Add num_entities
        self.num_entities = len(env.recipe_book.entities)

    def observation(self, observation):
        # Convert everything to torch tensors
        for k in observation:
            dtype = torch.long if '_index' in k else torch.float
            observation[k] = torch.tensor(observation[k], dtype=dtype)
        return DotDict(observation)


class Environment:
    def __init__(self, flags, gym_env):
        self.gym_env = gym_env
        self.env_name = flags.env.lower()
        self.episode_return = None
        self.episode_step = None
        self.initial_seed = flags.seed or gym_env.seed

    def reset_env(self):
        return self.gym_env.reset()

    def initial(self):
        initial_reward = torch.zeros(1, 1)
        initial_last_action = torch.zeros(1, 1, dtype=torch.int64)
        self.episode_return = torch.zeros(1, 1)
        self.episode_step = torch.zeros(1, 1, dtype=torch.int32)
        initial_done = torch.ones(1, 1, dtype=torch.bool)

        gym_env_state = self.reset_env()

        formatted_obs_dict = {k: _format_env_output(gym_env_state[k]) for k in gym_env_state}

        result_dict = dict(
            reward=initial_reward,
            done=initial_done,
            episode_return=self.episode_return,
            episode_step=self.episode_step,
            last_action=initial_last_action,
            **formatted_obs_dict,
        )

        return result_dict

    def step(self, action):
        gym_env_state, reward, done, unused_info = self.gym_env.step(action.item())
        self.episode_step += 1
        self.episode_return += reward
        episode_step = self.episode_step
        episode_return = self.episode_return
        if done:
            gym_env_state = self.reset_env()
            self.episode_return = torch.zeros(1, 1)
            self.episode_step = torch.zeros(1, 1, dtype=torch.int32)

        reward = torch.tensor(reward).view(1, 1)
        done = torch.tensor(done).view(1, 1)

        formatted_obs_dict = {k: _format_env_output(gym_env_state[k]) for k in gym_env_state}

        result_dict = dict(
            reward=reward,
            done=done,
            episode_return=episode_return,
            episode_step=episode_step,
            last_action=action,
            **formatted_obs_dict,
        )

        return result_dict

    def eval(self, split):
        self.gym_env.eval(split)

    def close(self):
        self.gym_env.close()
