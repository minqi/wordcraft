import sys

import torch
import torch.nn as nn


def is_int_str(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False


class Human(nn.Module):
	def forward(self, obs):
		return self._get_stdin_action(obs)

	def _get_stdin_action(self, obs):
		action = sys.stdin.readline()
		if is_int_str(action):
			table_index = obs['table_index']
			i = int(action) - 1
			if i < len(table_index):
				return i
		
		return self._get_stdin_action(obs)
		