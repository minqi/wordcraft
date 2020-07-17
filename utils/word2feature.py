import os

import numpy as np
from gym.utils import seeding
import spacy


class FeatureMap:
	"""
	Used for initializing entity features.
	"""
	def __init__(
		self,
		words,
		feature_type='glove',
		random_feature_size=300, # only for random features
		random_feature_std=0.4,
		shuffle=False,
		seed=None,
		savedir='~/model_cache/spacy',
		cache_id='latest'
	):
		self.feature_map = {}
		self.feature_dim = 0
		self.np_random, self.seed = seeding.np_random(seed)
		self.savedir = savedir
		self.cache_id = cache_id

		# Initialize featurizer
		if feature_type == 'one_hot':
			one_hots = np.eye(len(words))
			for i, e in enumerate(words):
				self.feature_map[e] = one_hots[i]
			self.feature_dim = one_hots[i].shape[-1]

		elif feature_type == 'random':
			b = random_feature_std/np.sqrt(2)
			rand_features = self.np_random.laplace(
				loc=0, scale=b, size=(len(words), random_feature_size)
			)
			for i, e in enumerate(words):
				self.feature_map[e] = rand_features[i,:]
			self.feature_dim = rand_features[i,:].shape[-1]

		elif feature_type == 'glove':
			loaded = self._load_feature_map()
			if loaded is not None:
				words, feature_vectors = loaded
			else:
				print('Loading glove model...')
				word2vec = spacy.load('en_core_web_lg') # 300-dim GloVe embeddings
				print('Loaded glove model.')
				feature_vectors = []
				for e in words:
					tokens_embeddings = []
					for token in e.split():
						tokens_embeddings.append(word2vec(token).vector)
					feature_vectors.append(np.array(tokens_embeddings).max(0))

				if cache_id:
					feature_vectors = np.array(feature_vectors)
					self._save_feature_map(words, feature_vectors)

			if shuffle:
				self.np_random.shuffle(feature_vectors)
			self.feature_map = dict(zip(words, feature_vectors))
			self.feature_dim = self.feature_map[words[0]].shape[-1]

		else:
			raise ValueError(f'Feature type {feature_type} is not implemented.')

	def seed(self, seed):
		self.np_random, self.seed = seeding.np_random(seed)

	def feature(self, word):
		return self.feature_map[word]

	def _basepath(self, savedir, cache_id):
		basepath = os.path.expandvars(os.path.expanduser(savedir))
		basepath = os.path.join(basepath, cache_id)
		return basepath

	def _save_feature_map(self, vocab, feature_vectors, savedir=None, cache_id=None):
		if savedir is None:
			savedir = self.savedir

		if cache_id is None:
			cache_id = self.cache_id

		basepath = self._basepath(savedir, cache_id)
		if not os.path.exists(basepath):
			os.makedirs(basepath, exist_ok=True)
		feature_bin_path = os.path.join(basepath, f'{self.cache_id}')
		vocab_path = os.path.join(basepath, f'{self.cache_id}.vocab')

		np.save(feature_bin_path, feature_vectors)

		with open(vocab_path, 'w+') as vocab_file:
			for w in vocab:
				vocab_file.write(w+'\n')

	def _load_feature_map(self, savedir=None, cache_id=None):
		if savedir is None:
			savedir = self.savedir

		if cache_id is None:
			cache_id = self.cache_id

		basepath = self._basepath(savedir, cache_id)
		feature_bin_path = os.path.join(basepath, f'{self.cache_id}.npy')
		vocab_path = os.path.join(basepath, f'{self.cache_id}.vocab')

		if os.path.exists(feature_bin_path) and os.path.exists(vocab_path):
			with open(vocab_path) as vocab_file:
				words = [w.strip() for w in vocab_file]
			feature_vectors = np.load(feature_bin_path)
			feature_map = dict(zip(words, feature_vectors))
			return words, feature_vectors
		else:
			return None