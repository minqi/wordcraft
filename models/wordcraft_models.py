from enum import IntEnum

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import DeviceAwareModule


# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def initialize_parameters(m):
	classname = m.__class__.__name__
	if classname.find('Linear') != -1:
		m.weight.data.normal_(0, 1)
		m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
		if m.bias is not None:
			m.bias.data.fill_(0)


def minmax_features(features):
	stacked_features = torch.stack(features, -1)
	min_vector,_ = stacked_features.min(-1)
	max_vector,_ = stacked_features.max(-1)
	return torch.cat((min_vector, max_vector), -1)


class RelationType(IntEnum):
	COMBINES_WITH = 0
	COMPONENT_OF = 1


class SelfAttentionACNet(DeviceAwareModule):
	def __init__(
		self, 
		observation_space, action_space, 
		arch='selfattn',
		key_size=128,
		value_size=128,
		hidden_size=128,
		use_minmax_features=False,
		sum_relation_inputs=False,
		kg_model=None,
		use_kg_goal_score=False,
		use_kg_selection_score=False,
		use_kg_only=False,
		rb=None):
		super().__init__()

		self.rb = rb

		table_size = observation_space['table_index'].shape[-1]
		goal_feature_size = observation_space['goal_features'].shape[-1]
		self.selection_size, selection_feature_size = observation_space['selection_features'].shape
		assert goal_feature_size == selection_feature_size, 'Goal and selection feature sizes must match.'

		self.num_actions = table_size

		self.arch = arch
		self.kg_model = kg_model
		if self.kg_model:
			self.kg_model.to(self.device)

		self.use_kg_goal_score = use_kg_goal_score
		self.use_kg_selection_score = use_kg_selection_score
		self.use_kg_only = use_kg_only

		self.key_size = key_size
		self.value_size = value_size
		self.hidden_size = hidden_size

		self.use_minmax_features = use_minmax_features
		self.sum_relation_inputs = sum_relation_inputs

		self.input_feature_size = goal_feature_size

		# Predict queries from goal and selection features
		if use_minmax_features:
			self.input_size = self.input_feature_size*2

		if arch == 'relational_selfattn':
			# Use a the same MLP to learn pairwise features between goals and selection (à la Relational Networks)
			if not use_minmax_features:
				if self.sum_relation_inputs:
					self.input_size = self.input_feature_size
				else:
					self.input_size = self.input_feature_size*2
			self.rel_fc = nn.Linear(self.input_size, hidden_size)
			self.q = nn.Linear(hidden_size, key_size)

		elif arch == 'selfattn':
			# Concat goal feature with selection features
			if not use_minmax_features:
				self.input_size = (1 + self.selection_size)*self.input_feature_size
			self.q = nn.Linear(self.input_size, key_size)

		else:
			raise ValueError(f'Unsupported arch {arch}.')

		self.k = nn.Linear(self.input_feature_size, key_size)
		self.v = nn.Linear(self.input_feature_size, value_size)

		self.fc_mixture_weights = nn.Linear(value_size, 3)
		self.policy = nn.Linear(value_size, self.num_actions)
		self.baseline = nn.Linear(value_size, 1)


	def initial_state(self, batch_size):
		return tuple()

	def kg_score_pairwise_relations(self, obj_idx, subj_idx_list, rel_enum):
		with torch.no_grad():
			num_s = subj_idx_list.shape[-1]

			o = obj_idx.repeat(1, num_s).flatten(0, 1)
			o[o < 0] = 0
			o_mask = obj_idx.flatten() >= 0 # T*B x 1 batch indices that need to be zeroed 

			r = torch.zeros_like(o, dtype=int).to(self.device)
			r[:] = rel_enum

			subj_idx_list_shape = subj_idx_list.shape
			s = subj_idx_list.clone()
			s_mask = s >= 0
			s[~s_mask] = 0
			s = s.flatten(0, 1)

			scores = self.kg_model.score(r, s, o).view(subj_idx_list_shape)

			scores[~o_mask,:] = s_mask.float()[~o_mask,:]
			scores = scores + (s_mask.float() + 1e-45).log()

		return scores

	def forward(self, inputs, core_state=(), greedy=False, **kwargs):
		goal_features = inputs['goal_features']
		T, B, *_ = goal_features.shape

		goal_features = goal_features.flatten(0, 1)
		goal_index = inputs['goal_index'].flatten(0, 1)
		selection_features = inputs['selection_features'].flatten(0, 1)
		selection_index = inputs['selection_index'].flatten(0, 1)
		table_features = inputs['table_features'].flatten(0, 1)
		table_index = inputs['table_index'].flatten(0, 1)

		input_features = [goal_features,] + [selection_features[:,i,:] for i in range(self.selection_size)]
		if self.arch == 'relational_selfattn':
			x = torch.zeros(T*B, self.hidden_size).to(goal_features.device)
			for i in range(1, len(input_features)):
				if not self.use_minmax_features:
					if self.sum_relation_inputs:
						in_features = input_features[0] + input_features[i]
					else:
						in_features = torch.cat((input_features[0], input_features[i]), -1)
				else:
					in_features = minmax_features(input_features[0],input_features[i])
				x = x + F.relu(self.rel_fc(in_features))
		else:
			if not self.use_minmax_features:
				x = torch.cat(input_features, -1)
			else:
				x = minmax_features(input_features)

		query = self.q(x).unsqueeze(-2) # T*B x 1 x d_key
		keys = self.k(table_features).transpose(1,2).contiguous() # T*B x d_key x table_size
		self_attn = torch.bmm(query, keys)/np.sqrt(self.key_size) # T*B x 1 x table_size
		policy_logits = self_attn.squeeze(-2)

		values = torch.bmm(self_attn, self.v(table_features)).squeeze(1)

		if self.kg_model:
			goal_scores = 0
			selection_scores = 0

			if self.use_kg_goal_score:
				goal_scores = self.kg_score_pairwise_relations(goal_index, table_index, RelationType.COMPONENT_OF)
			if self.use_kg_selection_score:
				# if selection_index[0, 0] >= 0:
				# 	print('on hand: ', self.rb.entities[selection_index[0,0]])
				# else:
				# 	print('empty handed...')
				selection_scores = self.kg_score_pairwise_relations(selection_index[:,0], table_index, RelationType.COMBINES_WITH)

			if self.use_kg_only:
				policy_logits = policy_logits*0
				policy_logits = policy_logits + goal_scores + selection_scores
			else:
				mixture_weights = F.softmax(self.fc_mixture_weights(values), dim=1).unsqueeze(-1)		
				policy_logits = mixture_weights[:,0]*policy_logits + mixture_weights[:,1]*goal_scores + mixture_weights[:,2]*selection_scores

		if greedy:
			action = torch.argmax(policy_logits, dim=1)
		else:
			action = torch.multinomial(F.softmax(policy_logits, dim=-1), num_samples=1)

		# if table_index.shape[0] == 1:
		# 	print(self.rb.entities[goal_index.flatten()], [self.rb.entities[i] for i in table_index.flatten()], F.softmax(policy_logits, dim=-1))
		# 	print('selected action', action, self.rb.entities[table_index[0][action]])

		# values = torch.bmm(self_attn, self.v(table_features)).squeeze(1) # T*B x 1 x d_value
		baseline = self.baseline(values) 

		policy_logits = policy_logits.view(T, B, self.num_actions)
		baseline = baseline.view(T, B)
		action = action.view(T, B)

		return dict(policy_logits=policy_logits, baseline=baseline, action=action), core_state


class SimpleACNet(DeviceAwareModule):
	def __init__(
		self, 
		observation_space, action_space, arch, hidden_size=300):
		super().__init__()

		self.num_actions = action_space.n
		self.arch = arch

		if arch == 'mlp_embed':
			vocab_size = observation_space['table_index'].nvec[0] + 1 # handle -1 placeholder
			self.embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
			embedding_size = self.embeddings.embedding_dim
		else:
			embedding_size = observation_space['table_features'].shape[1]

		table_size = observation_space['table_index'].shape[0]
		selection_size = observation_space['selection_index'].shape[0] - 1 # Don't care about last selected item per round

		max_active_entities = (table_size + selection_size + 1)
		input_size = embedding_size*max_active_entities

		self.policy = nn.Sequential(
			nn.Linear(input_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, self.num_actions)
		)
		self.baseline = nn.Linear(input_size, 1)

	def initial_state(self, batch_size):
		return tuple()

	def forward(self, inputs, core_state=(), greedy=False, **kwargs):
		goal_features = inputs['goal_features']
		T, B, *_ = goal_features.shape

		if self.arch == 'mlp_embed':
			goal_index = inputs['goal_index'].flatten(0, 1) + 1
			selection_index = inputs['selection_index'].flatten(0, 1) + 1
			table_index = inputs['table_index'].flatten(0, 1) + 1

			input_indices = torch.cat((goal_index, selection_index[:,:-1], table_index), dim=-1)
			input_features = self.embeddings(input_indices).flatten(1, 2)
		else:
			goal_features = goal_features.flatten(0, 1)
			selection_features = inputs['selection_features'].flatten(0, 1)
			table_features = inputs['table_features'].flatten(0, 1)

			input_features = torch.cat((goal_features.unsqueeze(1), selection_features[:,:-1], table_features), dim=1).flatten(1, 2)

		policy_logits = self.policy(input_features)
		baseline = self.baseline(input_features)

		if greedy:
			action = torch.argmax(policy_logits, dim=-1)
		else:
			action = torch.multinomial(F.softmax(policy_logits, dim=-1), num_samples=1)
		
		policy_logits = policy_logits.view(T, B, self.num_actions)
		baseline = baseline.view(T, B)
		action = action.view(T, B)

		return dict(policy_logits=policy_logits, baseline=baseline, action=action), core_state
