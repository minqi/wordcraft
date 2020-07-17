import os
from enum import IntEnum

import numpy as np
import gym
from gym.utils import seeding

from utils import seed as utils_seed
from utils.word2feature import FeatureMap
from wordcraft.recipe_book import Recipe, RecipeBook


NO_RECIPE_PENALTY = -0.1
IRRELEVANT_RECIPE_PENALTY = -0.1
GOAL_REWARD = 1.0
SUBGOAL_REWARD = 1.0

class WordCraftEnv(gym.Env):
	"""
	Simple text-only RL environment for crafting multi-step recipes.

	At a high level, the state consists of a goal, the inventory, and the current selection.
	"""
	def __init__(
		self,
		data_path='datasets/alchemy2.json',
		recipe_book_path=None,
		feature_type='glove',
		shuffle_features=False,
		random_feature_size=300,
		max_depth=1,
		split='by_recipe',
		train_ratio=1.0,
		num_distractors=0,
		uniform_distractors=False,
		max_mix_steps=1,
		subgoal_rewards=True,
		seed=None,
	):
		super().__init__()

		self.eval_mode = False

		if seed is None:
			seed = int.from_bytes(os.urandom(4), byteorder="little")
		self.set_seed(seed)
		utils_seed(seed)

		if recipe_book_path is not None:
			self.recipe_book = RecipeBook.load(recipe_book_path)
			self.recipe_book.set_seed(seed)
			max_depth = self.recipe_book.max_depth
		else:
			self.recipe_book = RecipeBook(
				data_path=data_path, max_depth=max_depth, split=split, train_ratio=train_ratio, seed=seed)

		self.feature_map = FeatureMap(
			words=self.recipe_book.entities,
			feature_type=feature_type, random_feature_size=random_feature_size,
			shuffle=shuffle_features,
			seed=seed)

		self.max_selection_size = self.recipe_book.max_recipe_size
		self.max_mix_steps = max(max_mix_steps or max_depth, max_depth)
		self.max_steps = self.max_selection_size*self.max_mix_steps

		self.sample_depth = max_depth

		self.subgoal_rewards = subgoal_rewards
		self.max_depth = max_depth
		self.num_distractors = num_distractors
		self.uniform_distractors = uniform_distractors

		self.max_table_size = 2**max_depth + num_distractors + self.max_mix_steps

		self.task = None
		self.distractors = []
		self.goal_features = np.zeros(self.feature_map.feature_dim)

		self._reset_table()
		self._reset_selection()
		self._reset_history()

		self.episode_step = 0
		self.episode_mix_steps = 0
		self.episode_reward = 0
		self.done = False

		obs = self.reset()
		num_entities = len(self.recipe_book.entities)
		dspaces = {
			'goal_index': gym.spaces.MultiDiscrete([num_entities]),
			'goal_features': gym.spaces.Box(shape=self.goal_features.shape, low=-1., high=1.),
			'table_index': gym.spaces.MultiDiscrete(self.max_table_size*[num_entities]),
			'table_features': gym.spaces.Box(shape=self.table_features.shape, low=-1., high=1.),
			'selection_index': gym.spaces.MultiDiscrete(self.max_selection_size*[num_entities]),
			'selection_features': gym.spaces.Box(shape=self.selection_features.shape, low=-1., high=1.),
		}
		self.observation_space = gym.spaces.Dict(dspaces)
		self.action_space = gym.spaces.Discrete(self.max_table_size) # Actions correspond to choosing an entity in a table position

	def reset(self):
		self.episode_step = 0
		self.episode_mix_steps = 0
		self.episode_reward = 0
		self.done = False

		self.task = self.recipe_book.sample_task(depth=self.sample_depth)
		self.distractors = self.recipe_book.sample_distractors(self.task, self.num_distractors, uniform=self.uniform_distractors)
		self.goal_features = self.feature_map.feature(self.task.goal)
		self._reset_selection()
		self._reset_table()
		self._reset_history()

		return self._get_observation()

	def eval(self, split='test'):
		self.eval_mode = True
		self.recipe_book.test_mode = (split == 'test')

	def train(self):
		self.eval_mode = False
		self.recipe_book.test_mode = False

	def set_seed(self, seed):
		self.np_random, self.seed = seeding.np_random(seed)

	def sample_depth(self, depth):
		self.sample_depth = depth

	def __max_table_size_for_depth(self, depth):
		return 2**depth - 1

	def _reset_table(self):
		if self.task:
			self.table = list(self.task.base_entities + self.distractors)
			self.np_random.shuffle(self.table)
		else:
			self.table = []
		self.table_index = -np.ones(self.max_table_size, dtype=int)
		self.table_features = np.zeros((self.max_table_size, self.feature_map.feature_dim))

		num_start_items = len(self.table)
		self.table_index[:num_start_items] = \
			np.array([self.recipe_book.entity2index[e] for e in self.table], dtype=int)
		if self.task:
			self.table_features[:num_start_items,:] = \
				np.array([self.feature_map.feature(e) for e in self.table])

	def _reset_selection(self):
		self.selection = []
		self.selection_index = -np.ones(self.max_selection_size, dtype=int)
		self.selection_features = np.zeros((self.max_selection_size, self.feature_map.feature_dim))

	def _reset_history(self):
		self.subgoal_history = set()

	def _get_observation(self):
		"""
		Note, includes indices for each inventory and selection item,
		since torchbeast stores actions in a shared_memory tensor shared among actor processes
		"""
		return {
			'goal_index': [self.recipe_book.entity2index[self.task.goal]],
			'goal_features': self.goal_features,
			'table_index': self.table_index,
			'table_features': self.table_features,
			'selection_index': self.selection_index,
			'selection_features': self.selection_features,
		}

	def step(self, action):
		reward = 0
		if self.done: # no-op if env is done
			return self._get_observation(), reward, self.done, {}

		# Handle invalid actions
		invalid_action = not (0 <= action < self.max_table_size)
		if invalid_action:
			self.episode_step += 1
			if self.episode_step >= self.max_steps:
				self.done = True

		i = self.table_index[action]
		e = self.recipe_book.entities[i]

		selection_size = len(self.selection)
		if selection_size < self.max_selection_size:
			# Update selection
			self.selection.append(e)
			self.selection_index[selection_size] = i
			self.selection_features[selection_size,:] = self.feature_map.feature(e)
			selection_size = len(self.selection)

		if selection_size == self.max_selection_size:
			self.episode_mix_steps += 1

			# Evaluate selection
			recipe = Recipe(self.selection)
			result = self.recipe_book.evaluate_recipe(recipe)

			if result is None:
				reward = NO_RECIPE_PENALTY if not self.eval_mode else 0

			elif result == self.task.goal:
				reward = GOAL_REWARD
				self.done = True
			elif result in self.task.intermediate_entities:
				reward = 0
				if result not in self.subgoal_history:
					self.subgoal_history.add(result)
					reward = SUBGOAL_REWARD if self.subgoal_rewards and not self.eval_mode else 0
			else:
				reward = IRRELEVANT_RECIPE_PENALTY if not self.eval_mode else 0
			self.episode_reward += reward

			if result:
				result_i = self.recipe_book.entity2index[result]
				table_size = len(self.table)
				self.table.append(result)
				self.table_index[table_size] = result_i
				self.table_features[table_size,:] = self.feature_map.feature(result)

			# Clear selection
			self._reset_selection()

		self.episode_step += 1
		if self.episode_mix_steps >= self.max_mix_steps or self.episode_step >= self.max_steps:
			self.done = True

		obs = self._get_observation()

		return obs, reward, self.done, {}

	def _display_ascii(self, mode='human'):
		"""
		Render the env state as ascii:

		Combine the ingredients to make *torch*

		-------------------------------------------------------
		1:fire, 2:wind, 3:sand, 4:star, 5:wood, 6:stick, 7:coal
		-------------------------------------------------------

		(on hand): stick

		Subgoal rewards: 0
		"""
		goal_str = f'Combine the ingredients to make *{self.task.goal}*'
		if mode == 'human':
			table_str = f"{', '.join([f'{i+1}:{e}' for i, e in enumerate(self.table)])}"
		else:
			table_str = f"{', '.join(self.table)}"
		selection_str = f"(on hand): {', '.join(self.selection)}"
		hr = ''.join(['-']*50)

		# output = f'\n{goal_str}\n\n{hr}\n{table_str}\n{hr}\n\n{selection_str}\n\nSubgoal rewards: {self.episode_reward}\n'
		output = f'\n{goal_str}\n\n{hr}\n{table_str}\n{hr}\n\n{selection_str}\n\n'

		print(output)

	def render(self, mode='human'):
		self._display_ascii(mode)


gym.envs.registration.register(
    id='wordcraft-multistep-goal-v0',
    entry_point=f"{__name__}:WordCraftEnv",
)
