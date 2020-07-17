import numpy as np
import torch
import torch.nn as nn

from utils import DeviceAwareModule


class RandomAgent(DeviceAwareModule):
	def __init__(self, observation_space, action_space):
		super().__init__()

		self.action_space = action_space
		self.num_actions = action_space.n

		goal_shape = np.prod(observation_space['goal_features'].shape)
		self.random_baseline = nn.Linear(goal_shape, 1)

	def initial_state(self, batch_size):
		return tuple()

	def forward(self, inputs, core_state=(), **kwargs):
		goal_idx = inputs['goal_index']
		goal_features = inputs['goal_features'].flatten(0, 1)
		table_idx = inputs['table_index'].flatten(0, 1)
		T, B, *_ = goal_idx.shape

		core_state = self.initial_state(B)

		policy_logits = torch.log(torch.ones(T, B, self.num_actions)/self.num_actions).to(self.device)
		baseline = self.random_baseline(goal_features).view(T, B)
		action = torch.tensor([self.action_space.sample() for _ in range(T*B)]).view(T, B)

		return dict(policy_logits=policy_logits, baseline=baseline, 
					action=action), core_state
