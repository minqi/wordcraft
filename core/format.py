from __future__ import absolute_import

import os
import json
import numpy as np
import re
import torch

import babyai
from babyai import utils


def get_vocab_path(model_name):
    return os.path.join(utils.get_model_dir(model_name), "vocab.json")


class Vocabulary:
    def __init__(self, model_name):
        self.path = get_vocab_path(model_name)
        self.max_size = 100
        if os.path.exists(self.path):
            self.vocab = json.load(open(self.path))
        else:
            self.vocab = {}

    def __getitem__(self, token):
        if not (token in self.vocab.keys()):
            if len(self.vocab) >= self.max_size:
                raise ValueError("Maximum vocabulary capacity reached")
            self.vocab[token] = len(self.vocab) + 1
        return self.vocab[token]

    def save(self, path=None):
        if path is None:
            path = self.path
        utils.create_folders_if_necessary(path)
        json.dump(self.vocab, open(path, "w"))

    def copy_vocab_from(self, other):
        '''
        Copy the vocabulary of another Vocabulary object to the current object.
        '''
        self.vocab.update(other.vocab)


class InstructionsPreprocessor(object):
    def __init__(self, model_name, max_length=None, load_vocab_from=None):
        self.model_name = model_name
        self.vocab = Vocabulary(model_name)
        self.max_length = max_length

        path = get_vocab_path(model_name)
        if not os.path.exists(path) and load_vocab_from is not None:
            # self.vocab.vocab should be an empty dict
            secondary_path = get_vocab_path(load_vocab_from)
            if os.path.exists(secondary_path):
                old_vocab = Vocabulary(load_vocab_from)
                self.vocab.copy_vocab_from(old_vocab)
            else:
                raise FileNotFoundError('No pre-trained model under the specified name')

    def process_instr(self, s):
        tokens = re.findall("([a-z]+)", s.lower())
        array = np.array([self.vocab[token] for token in tokens])
        length = len(array)
        if self.max_length is not None:
            padded_array = np.zeros(self.max_length)
            max_index = min(self.max_length, len(array))
            padded_array[:max_index] = array[:max_index]
            array = padded_array
            
        return array, length

    def __call__(self, obss, device=None):
        raw_instrs = []
        max_instr_len = 0
        instr_lengths = np.zeros((len(obss)))

        if isinstance(obss, dict):
            instr, instr_length = self.process_instr(obss["mission"])
            instr = torch.tensor(instr, device=device, dtype=torch.long)
            return instr, instr_length
        else:
            for i, obs in enumerate(obss):
                instr, length = self.process_instr(obs["mission"])
                raw_instrs.append(instr)
                max_instr_len = max(len(instr), max_instr_len)
                instr_lengths[i] = length

            instrs = np.zeros((len(obss), max_instr_len))
        
            for i, instr in enumerate(raw_instrs):
                instrs[i, :len(instr)] = instr

            instrs = torch.tensor(instrs, device=device, dtype=torch.long)
            instr_lengths = torch.tensor(instr_lengths, device=device, dtype=torch.long)
            return instrs, instr_lengths


class RawImagePreprocessor(object):
    def __call__(self, obss, device=None):
        if isinstance(obss, dict) and "image" in obss:
            images = obss['image']
        elif isinstance(obss, np.ndarray):
            return torch.tensor(obss)
        else:
            images = np.array([obs["image"] for obs in obss])

        images = torch.tensor(images, device=device, dtype=torch.float)
        return images


class IntImagePreprocessor(object):
    def __init__(self, num_channels, max_high=255):
        self.num_channels = num_channels
        self.max_high = max_high
        self.offsets = np.arange(num_channels) * max_high
        self.max_size = int(num_channels * max_high)

    def __call__(self, obss, device=None):
        if isinstance(obss, dict):
            images = obss['image']
        else:
            images = np.array([obs["image"] for obs in obss])
        # The padding index is 0 for all the channels
        images = (images + self.offsets) * (images > 0)
        images = torch.tensor(images, device=device, dtype=torch.long)
        return images


class ObssPreprocessor:
    def __init__(self, model_name, obs_space, instr_max_length=None, load_vocab_from=None):
        self.image_preproc = RawImagePreprocessor()
        self.obs_space = {
            "image": obs_space.shape,
        }

        if instr_max_length is not None and instr_max_length > 0:
            self.instr_preproc = InstructionsPreprocessor(model_name, instr_max_length, load_vocab_from)
            self.vocab = self.instr_preproc.vocab
            self.obs_space["instr"] = self.vocab.max_size

    def __call__(self, obss, device=None):
        obs_ = babyai.rl.DictList()

        if "image" in self.obs_space.keys():
            obs_.image = self.image_preproc(obss, device=device)

        if "instr" in self.obs_space.keys():
            obs_.instr, obs_.instr_length = self.instr_preproc(obss, device=device)

        return obs_


class IntObssPreprocessor(object):
    def __init__(self, model_name, obs_space, instr_max_length, load_vocab_from=None):
        image_obs_space = obs_space.spaces["image"]
        self.image_preproc = IntImagePreprocessor(image_obs_space.shape[-1],
                                                  max_high=image_obs_space.high.max())
        self.obs_space = {
            "image": self.image_preproc.max_size,
        }

        if instr_max_length is not None and instr_max_length > 0:
            self.instr_preproc = InstructionsPreprocessor(model_name, instr_max_length, load_vocab_from)
            self.vocab = self.instr_preproc.vocab
            self.obs_space["instr"] = self.vocab.max_size,

    def __call__(self, obss, device=None):
        obs_ = babyai.rl.DictList()

        if "image" in self.obs_space.keys():
            obs_.image = self.image_preproc(obss, device=device)

        if "instr" in self.obs_space.keys():
            obs_.instr, obs_.instr_length = self.instr_preproc(obss, device=device)

        return obs_