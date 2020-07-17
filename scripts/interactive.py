import argparse

import torch
import gym

from wordcraft.recipe_book import RecipeBook
from wordcraft.env import WordCraftEnv
from models import Human


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Interactive WordCraft env")
	parser.add_argument("--depth", type=int, default=1, help='Task depth')
	parser.add_argument("--max_mix_steps", type=int, default=2, help='Max tries at combining ingredients')
	parser.add_argument("--num_distractors", type=int, default=1, help='Number of distractors')
	parser.add_argument("--seed", type=int, default=40, help='Max tries at combining ingredients')
	args = parser.parse_args()

	max_mix_steps = args.max_mix_steps if args.max_mix_steps > 0 else args.depth

	env = gym.make(
		'wordcraft-multistep-goal-v0', 
		max_depth=args.depth, 
		max_mix_steps=max_mix_steps, 
		num_distractors=args.num_distractors, 
		seed=args.seed)
	agent = Human()

	obs = env.reset()
	while True:
		env.render()
		obs, reward, done, info = env.step(agent(obs))
		if done:
			print(f'Episode ended with reward {reward}\n')
			obs = env.reset()