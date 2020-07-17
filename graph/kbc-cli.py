import os
import sys
import argparse
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from utils import seed as utils_seed
from wordcraft.recipe_book import RecipeBook
from utils.word2feature import FeatureMap
from .kbcr import ComplEx
from .kbcr.regularizers import N2, N3
from graph.kbcr.evaluation import evaluate


def recipebook2relations(recipe_book, feature_map=None, device=None):
	"""
	Takes a recipe book and returns a set of train and test triplets (s, r, o).
	"""
	COMBINES_WITH = 0
	COMPONENT_OF = 1

	entity2index = recipe_book.entity2index

	def add_goal_recipe_relations_to_set(goal, recipe, s):
		if len(recipe.keys()) == 2:
			e1, e2 = recipe
		else:
			e1, e2 = list(recipe.keys())*2
		e1_idx = entity2index[e1]
		e2_idx = entity2index[e2]
		g_idx = entity2index[goal]
		s.update({
			(e1_idx, COMBINES_WITH, e2_idx),
			(e2_idx, COMBINES_WITH, e1_idx),
			(e1_idx, COMPONENT_OF, g_idx),
			(e2_idx, COMPONENT_OF, g_idx)})

	train_relations = set()
	test_relations = set()
	if recipe_book.split in ['by_goal_train_terminals', 'by_goal']:
		for goal in recipe_book.goals_train:
			if goal in recipe_book.entity2recipes:
				for recipe in recipe_book.entity2recipes[goal]:
					add_goal_recipe_relations_to_set(goal, recipe, train_relations)
		for goal in recipe_book.goals_test:
			if goal in recipe_book.entity2recipes:
				for recipe in recipe_book.entity2recipes[goal]:
					add_goal_recipe_relations_to_set(goal, recipe, test_relations)

	elif recipe_book.split in ('by_recipe_train_all_goals', 'by_recipe'):
		for recipe in recipe_book.recipes_train:
			goal = recipe_book.recipe2entity[recipe]
			add_goal_recipe_relations_to_set(goal, recipe, train_relations)

		for recipe in recipe_book.recipes_test:
			goal = recipe_book.recipe2entity[recipe]
			add_goal_recipe_relations_to_set(goal, recipe, test_relations)

	else:
		raise ValueError(f'Unsupported split {rb.split}')

	if not device:
		device = torch.device('cpu')

	train_relations = torch.tensor(np.array(list(train_relations)), dtype=int).to(device)
	test_relations = torch.tensor(np.array(list(test_relations)), dtype=int).to(device)

	return train_relations, test_relations


class RelationsDataset(Dataset):
	def __init__(self, relations, feature_type='integer', recipe_book=None):
		if feature_type == 'glove':
			assert recipe_book is not None, 'Generating GloVe features requires a recipe book.'
		
		self.relations = relations

		self.feature_type = feature_type
		self.recipe_book = recipe_book
		self.num_entities = len(self.recipe_book.entities)
		self.num_predicates = torch.max(relations[:,1]).item() + 1
		self.feature_size = 1

		device = relations.device

		if self.feature_type == 'integer':
			self.relations = relations

		if feature_type == 'glove':
			self.feature_map = FeatureMap(words=recipe_book.entities, feature_type='glove')
			num_relations = len(relations)
			self.feature_size = self.feature_map.feature_dim

			# precompute index2features here
			self.index2features = torch.tensor(
				[self.feature_map.feature(e) for e in self.recipe_book.entities], 
				dtype=torch.float).to(device)

			self.subjects = torch.zeros((num_relations, self.feature_size), dtype=torch.float).to(device)
			self.objects = torch.zeros_like(self.subjects, dtype=torch.float).to(device)
			self.predicates = torch.zeros((num_relations,), dtype=torch.long).to(device)

			for i in range(num_relations):
				s_idx, p_idx, o_idx = relations[i,:]
				s_word = recipe_book.entities[s_idx]
				o_word = recipe_book.entities[o_idx]
				self.subjects[i,:] = torch.from_numpy(self.feature_map.feature(s_word))
				self.objects[i,:] = torch.from_numpy(self.feature_map.feature(o_word))
				self.predicates[i] = p_idx

	def feature_from_index(self, index):
		"""
		Input: index, a 1-D tensor or array of indices
		"""
		return self.index2features[index]

	def __len__(self):
		return len(self.relations)

	def __getitem__(self, idx):
		indices = (self.relations[idx,0], self.relations[idx, 1], self.relations[idx, 2])
		if self.feature_type == 'integer':
			return indices, indices
		elif self.feature_type == 'glove':
			return (self.subjects[idx,:], self.predicates[idx], self.objects[idx,:]), indices
		else:
			raise ValueError(f'Invalid feature_type {self.feature_type}')


def metrics_to_str(metrics):
	return f'MRR {metrics["MRR"]:.6f}\tH@1 {metrics["hits@1"]:.6f}\tH@3 {metrics["hits@3"]:.6f}\t' \
		f'H@5 {metrics["hits@5"]:.6f}\tH@10 {metrics["hits@10"]:.6f}'


def get_cmd_args():
	parser = argparse.ArgumentParser(description="PyTorch Scalable Agent")

	# Path params
	parser.add_argument("--xpid", type=str, default='latest',
						help="Experiment name or model ID.")
	parser.add_argument("--data_path", type=str, default="datasets/alchemy2.json",
						help="Path to recipe data.")
	parser.add_argument("--save_dir", type=str, default='~/model_cache/wordcraft',
						help="Path to save binary serialized RecipeBook instance and ComplEx models to preserve train/test split.")

	# Data params
	parser.add_argument("--max_depth", type=int, default=1,
						help="Max depth of recipes in split.")
	parser.add_argument("--split", type=str, default='by_recipe_train_all_goals', choices=['by_goal_train_terminals', 'by_recipe_train_all_goals', 'by_goal', 'by_recipe'],
						help="How tasks are split into train and test sets.")
	parser.add_argument("--train_ratio", type=float, default=0.5,
						help="Size ratio of train set to test set.")
	parser.add_argument("--feature_type", type=str, default='integer', choices=['integer', 'random', 'glove'],
						help="How entities are represented.")

	# KBC model params
	parser.add_argument("--embedding_size", type=int, default=300, 
						help="Size of KBC model embeddings.")

	# Training params
	parser.add_argument('--num_epochs', type=int, default=5)
	parser.add_argument('--batch_size', '-b', action='store', type=int, default=512)
	parser.add_argument('--learning_rate', '-l', action='store', type=float, default=0.1)
	parser.add_argument('--optimizer', '-o', action='store', type=str, default='adagrad', choices=['adagrad', 'adam', 'sgd'])
	parser.add_argument('--N2', action='store', type=float, default=None)
	parser.add_argument('--N3', action='store', type=float, default=None)
	parser.add_argument('--gradient-accumulation-steps', '--gas', action='store', type=int, default=1)
	parser.add_argument('--seed', action='store', type=int, default=0)

	parser.add_argument('--eval-batch-size', '-B', action='store', type=int, default=None)
	parser.add_argument('--validate_interval', '-V', action='store', type=int, default=10)

	# Logging
	parser.add_argument('--verbose', '-v', action='store_true')

	return parser.parse_args()


def main():
	args = get_cmd_args()

	logger = logging.getLogger(os.path.basename(sys.argv[0]))
	if args.verbose: logging.basicConfig(level = logging.INFO)

	utils_seed(args.seed)

	# Create recipe split + save split instance
	recipe_book = RecipeBook(
		data_path=args.data_path, 
		max_depth=args.max_depth, 
		split=args.split, 
		train_ratio=args.train_ratio)

	save_dir = os.path.expandvars(
		os.path.expanduser(os.path.join(args.save_dir, args.xpid)))
	if not os.path.exists(save_dir):
		os.makedirs(save_dir, exist_ok=True)
	recipe_save_path = os.path.join(save_dir, 'recipe_book.bin')

	recipe_book.save(recipe_save_path)
	
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# Set up training data
	train_relations, test_relations = recipebook2relations(recipe_book, device=device)
	if np.isclose(args.train_ratio, 1.0):
		test_relations = train_relations
	all_relations = torch.cat((train_relations, test_relations), dim=0).cpu()
	test_relations = test_relations.cpu()

	dataset = RelationsDataset(train_relations, feature_type=args.feature_type, recipe_book=recipe_book)
	dataloader = DataLoader(
		dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

	# Set up ComplEx
	num_entities = len(recipe_book.entities)
	num_predicates = dataset.num_predicates

	model = ComplEx(
		num_entities=num_entities, 
		num_predicates=num_predicates, 
		embedding_size=args.embedding_size,
		feature_type=args.feature_type,
		feature_size=dataset.feature_size,
		index2features=dataset.index2features if args.feature_type == 'glove' else None)
	model.to(device)

	# Set up optimizer and losses
	lr = args.learning_rate
	optimizer_factory = {
		'adagrad': lambda: optim.Adagrad(model.parameters(), lr=lr),
		'adam': lambda: optim.Adam(model.parameters(), lr=lr),
		'sgd': lambda: optim.SGD(model.parameters(), lr=lr)
	}
	optimizer = optimizer_factory[args.optimizer]()
	loss_function = nn.CrossEntropyLoss(reduction='mean')

	N2_weight = args.N2
	N3_weight = args.N3
	N2_reg = N2() if N2_weight is not None else None
	N3_reg = N3() if N3_weight is not None else None

	# Training loop
	validate_interval = args.validate_interval
	eval_batch_size = args.batch_size if args.eval_batch_size is None else args.eval_batch_size
	num_batches = len(dataloader)
	for epoch in range(args.num_epochs):
		epoch_loss_values = []
		batches = iter(dataloader)
		for batch_idx, batch in enumerate(batches):
			features_batch, index_batch = batch

			xs_batch, xp_batch, xo_batch = features_batch
			xsi_batch, xpi_batch, xoi_batch = index_batch

			sp_scores, po_scores, embeddings = model.forward(xp_batch, xs_batch, xo_batch)

			factors = [model.factor(e) for e in embeddings.values()]

			s_loss = loss_function(sp_scores, xoi_batch)
			o_loss = loss_function(po_scores, xsi_batch)

			loss = s_loss + o_loss

			loss += N2_weight * N2_reg(factors) if N2_weight is not None else 0.0
			loss += N3_weight * N3_reg(factors) if N3_weight is not None else 0.0

			if args.gradient_accumulation_steps > 1:
				loss = loss / args.gradient_accumulation_steps

			loss.backward()

			if batch_idx % args.gradient_accumulation_steps == 0 or batch_idx == num_batches:
				optimizer.step()
				optimizer.zero_grad()

			loss_value = loss.item()
			epoch_loss_values += [loss_value]

			logger.info(f'Epoch {epoch}/{args.num_epochs}\tBatch {batch_idx}/{num_batches}\tLoss {loss_value:.6f}')

		if validate_interval is not None and epoch % validate_interval == 0:
			metrics = evaluate(
						entity_embeddings=model.entity_embeddings, 
						predicate_embeddings=model.predicate_embeddings,
						all_triples=all_relations.data.numpy(),
						test_triples=test_relations.data.numpy(), 
						model=model, 
						batch_size=eval_batch_size,
						index2features=dataset.index2features if args.feature_type == 'glove' else None,
						device=device)
			logger.info(f'Epoch {epoch}/{args.num_epochs}\t"test" results\t{metrics_to_str(metrics)}')

	# Save ComplEx model to .tar
	torch.save(model.state_dict(), os.path.join(save_dir, f'complex.tar'))

if __name__ == '__main__':
	main()
