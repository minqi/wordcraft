import os
import json
import pickle
import collections
import random
import timeit
import copy

import numpy as np
from gym.utils import seeding

from utils.log import cprint


DEBUG = False


class Recipe(collections.Counter):
    """A hashable recipe.
    Allows for indexing into dictionaries.
    """
    def __hash__(self):
        return tuple(
                sorted(
                    self.items(),
                    key=lambda x: x[0] if x[0] is not None else '')).__hash__()

    def __len__(self):
        return len(list(self.elements()))


class Task:
    """
    A hashable recipe task.
    """
    def __init__(self, goal, base_entities, intermediate_entities, relevant_recipes):
        self.goal = goal
        self.base_entities = tuple(sorted(base_entities))
        self.intermediate_entities = tuple(sorted(intermediate_entities))
        self.relevant_recipes = tuple(relevant_recipes)

    def __hash__(self):
        return tuple((self.goal, self.base_entities, self.intermediate_entities, self.relevant_recipes)).__hash__()


class RecipeBook:
    def __init__(self, 
        data_path='datasets/alchemy2.json', max_depth=1, split=None, train_ratio=1.0, seed=None):
        self.test_mode = False
        self.train_ratio = train_ratio
        self.set_seed(seed)

        self._rawdata = self._load_data(data_path)
        self.max_depth = max_depth

        self.entities = tuple(self._rawdata['entities'].keys())
        self.entity2index = {e:i for i,e in enumerate(self.entities)}
        self.entity2recipes = collections.defaultdict(list)

        for e in self.entities:
            for r in self._rawdata['entities'][e]['recipes']:
                if e not in r:
                    self.entity2recipes[e].append(Recipe(r))
        self.entity2recipes = dict(self.entity2recipes)

        self.max_recipe_size = 0
        self.recipe2entity = collections.defaultdict(str)
        for entity, recipes in self.entity2recipes.items():
            for r in recipes:
                self.recipe2entity[r] = entity
                self.max_recipe_size = max(len(r), self.max_recipe_size)

        self.root_entities = set([e for e in self.entities if e not in self.entity2recipes])

        self.init_neighbors_combineswith()
        self.terminal_entities = set([e for e in self.entities if e not in self.neighbors_combineswith])

        self._init_tasks_for_depth(max_depth)
        self._init_recipe_weighted_entity_dist()

        self._init_data_split(split=split, train_ratio=train_ratio)

    def _random_choice(self, options):
        # Fast random choice
        i = self.np_random.randint(0, len(options))
        return options[i]

    def _load_data(self, path):
        f = open(path)
        jsondata = json.load(f)
        f.close()

        return jsondata

    def set_seed(self, seed):
        self.np_random, self.seed = seeding.np_random(seed)

    def save(self, path):
        """
        Serialize to bytes and save to file
        """
        path = os.path.expandvars(os.path.expanduser(path))
        f = open(path, 'wb+')
        pickle.dump(self, f)

    @staticmethod
    def load(path):
        """
        Returns a new RecipeBook object loaded from a binary file that is the output of save.
        """
        path = os.path.expandvars(os.path.expanduser(path))
        f = open(path, 'rb')
        return pickle.load(f)

    def get_recipes(self, entity):
        return self.entity2recipes[entity] if entity in self.entity2recipes else None

    def evaluate_recipe(self, recipe):
        e = self.recipe2entity[recipe]
        return e if e != '' else None

    def init_neighbors_combineswith(self):
        self.neighbors_combineswith = collections.defaultdict(set)
        for recipe in self.recipe2entity:
            e1, e2 = recipe if len(recipe.keys()) == 2 else list(recipe.keys())*2
            self.neighbors_combineswith[e1].add(e2)
            self.neighbors_combineswith[e2].add(e1)

    def sample_task(self, depth=None):
        """
        Returns a task tuple (<goal>, <intermediate entities>, <base entities>)
        """
        if depth is None:
            depth = self.np_random.choice(range(1,self.max_depth+1))

        sample_space = self.depth2task_test if self.test_mode and self.train_ratio < 1.0 else self.depth2task_train
        return self._random_choice(sample_space[depth])

    def sample_distractors(self, task, num_distractors=1, uniform=True):
        base_e = set(task.base_entities)
        intermediate_e = set(task.intermediate_entities)

        def is_valid(e):
            return e != task.goal and e not in base_e and e not in intermediate_e

        options = [(i, e) for i, e in enumerate(self.entities) if is_valid(e)]
        sample_index_space, sample_space = zip(*options)

        if uniform:
            return tuple(self._random_choice(sample_space, num_distractors).tolist())
        else:
            # sample according to recipe-weighted entity distribution
            sample_index_space = set(sample_index_space)
            dist = np.array([p for i, p in enumerate(self.entity_dist) if i in sample_index_space])
            dist /= dist.sum()
            return tuple(self.np_random.choice(sample_space, num_distractors, p=dist).tolist())

    def _generate_all_tasks_for_goal(self, goal, max_depth=3):
        base_entities = [goal]
        intermediate_entities = set()
        cprint(DEBUG,f'Expanding tasks to goal {goal}')
        self._expand_tasks_to_goal(goal, max_depth, base_entities, intermediate_entities)
        cprint(DEBUG,'Done.')

    def _expand_tasks_to_goal(self, goal, max_depth=1, base_entities=[], intermediate_entities=set(), relevant_recipes=[]):
        """
        DFS expansion of recipes for an entity to generate new tasks
        """
        for b in base_entities:
            if b not in self.root_entities: # Can't expand if it's a root entity or cyclic
                if b != goal: intermediate_entities.add(b)
                next_base_entities = base_entities[:]
                next_base_entities.remove(b)

                cur_depth = len(intermediate_entities) + 1

                cprint(DEBUG,'--Expanding base entity', b)

                # Expand each recipe for each base entity
                for recipe in self.entity2recipes[b]:
                    cprint(DEBUG,f'----Trying recipe for {b}, {recipe}')
                    expanded_entities = [e for e in recipe if e not in next_base_entities]
                    is_cycle = False
                    for e in recipe:
                        if e in intermediate_entities or e == goal: 
                            cprint(DEBUG,f'------Cycle detected, skipping recipe {recipe}')
                            is_cycle = True
                            break
                    if is_cycle:
                        continue

                    old_base_entities = next_base_entities
                    next_base_entities = expanded_entities + next_base_entities

                    # Add task
                    relevant_recipes.append(recipe)
                    task = Task(goal, next_base_entities, intermediate_entities, relevant_recipes[:])
                    if task not in self.depth2task[cur_depth]: 
                        self.depth2task[cur_depth].add(task)
                        cprint(DEBUG,f'------Adding task {task}')

                    if cur_depth < max_depth:
                        cprint(DEBUG,f'current depth is {cur_depth}')
                        self._expand_tasks_to_goal(goal, max_depth, next_base_entities, intermediate_entities, relevant_recipes[:])

                    relevant_recipes.remove(recipe)
                    next_base_entities = old_base_entities

                if b != goal: intermediate_entities.remove(b)

    def _init_tasks_for_depth(self, max_depth=2):
        self.depth2task = collections.defaultdict(set) # depth to task tuples

        total = 0
        for e in self.entities:
            # self._generate_all_tasks_for_goal(e)
            s = timeit.timeit(lambda: self._generate_all_tasks_for_goal(e, max_depth=max_depth), number=1)
            # print(f'Generated max-depth {max_depth} recipes for {e} in {s} s.')
            total += s

        print(f'Generated all max-depth {max_depth} tasks for {len(self.entities)} entities in {total} s.')

        for d in self.depth2task:
            self.depth2task[d] = tuple(self.depth2task[d])
            print(f"Depth {d} tasks: {len(self.depth2task[d])}")

    def _init_recipe_weighted_entity_dist(self):
        entities_cnt = dict({e: 0 for e in self.entities})
        for recipe in self.recipe2entity.keys():
            for e in recipe:
                entities_cnt[e] += 1

        unnormalized = np.array(list(entities_cnt.values())) + 1 # Even terminal entities have > 0 chance of being sampled 
        self.entity_dist = unnormalized/unnormalized.sum()

    def _init_data_split(self, split, train_ratio):
        self.split = split

        depths = range(1,self.max_depth+1)

        self.goals_train = []
        self.goals_test = []

        self.depth2task_train = {d:[] for d in depths}
        self.depth2task_test = {d:[] for d in depths}

        if split in ['debug', 'by_goal', 'by_goal_train_terminals']:
            # Map goals --> depth --> tasks
            self.goal2depth2task = {goal:{depth:[] for depth in depths} for goal in self.entities}
            for depth in self.depth2task:
                tasks = self.depth2task[depth]
                for task in tasks:
                    self.goal2depth2task[task.goal][depth].append(task)

            # Split goals into train and test
            all_goals = list(self.entities)
            self.np_random.shuffle(all_goals)
            if split == 'debug': train_ratio = 1.0
            train_size = int(np.ceil(train_ratio*len(all_goals)))

            if split == 'by_goal_train_terminals':
                assert train_size > len(self.terminal_entities), 'Train size must be > terminal entities'

                all_goals = list(set(all_goals) - self.terminal_entities)
                train_size = train_size - len(self.terminal_entities)

            self.goals_train = all_goals[:train_size]
            self.goals_test = all_goals[train_size:]

            if split == 'debug':
                self.goals_test = list(self.goals_train)

            for depth in depths:
                for goal in self.goals_train:
                    self.depth2task_train[depth] += (self.goal2depth2task[goal][depth])

                for goal in self.goals_test:
                    self.depth2task_test[depth] += (self.goal2depth2task[goal][depth])

        elif split in ['by_recipe', 'by_recipe_train_all_goals']:
            all_recipes = list(self.recipe2entity.keys())
            self.np_random.shuffle(all_recipes)
            train_size = int(np.ceil(train_ratio*len(all_recipes)))
            self.recipes_train = set(all_recipes[:train_size])
            self.recipes_test = set(all_recipes[train_size:])
            if split == 'by_recipe_train_all_goals':
                self._fill_recipe_entity_support()

            for depth in self.depth2task:
                tasks = self.depth2task[depth]
                for task in tasks:
                    is_test_task = False
                    for recipe in task.relevant_recipes:
                        if recipe in self.recipes_test:
                            self.depth2task_test[depth].append(task)
                            is_test_task = True
                            break
                    if not is_test_task: self.depth2task_train[depth].append(task)

        elif split == 'by_task':
            for depth in depths:
                all_tasks_at_depth = list(self.depth2task[depth])
                self.np_random.shuffle(all_tasks_at_depth)
                train_size_at_depth = int(np.ceil(train_ratio*len(all_tasks_at_depth)))

                self.depth2task_train[depth] = all_tasks_at_depth[:train_size_at_depth]
                self.depth2task_test[depth] = all_tasks_at_depth[train_size_at_depth:]

        else:
            raise ValueError(f'Unsupported split {split}')

        train_size = 0
        test_size = 0
        overlap = 0
        for depth in depths:
            train_tasks = set(self.depth2task_train[depth])
            test_tasks = set(self.depth2task_test[depth])

            train_size += len(train_tasks)
            test_size += len(test_tasks)

            overlap += len(train_tasks.intersection(test_tasks))

    def _fill_recipe_entity_support(self):
        # Make sure all entities are represented among self.recipes_train at depth=1 as either ingredient or goa
        def make_entity2recipes(recipes):
            entity2recipes = collections.defaultdict(set)
            for recipe in recipes:
                goal = self.recipe2entity[recipe]
                entity2recipes[goal].add(recipe)
                for e in recipe:
                    entity2recipes[e].add(recipe)
            return entity2recipes

        entity2recipes_train = make_entity2recipes(self.recipes_train)
        entity2recipes_test = make_entity2recipes(self.recipes_test)

        train_entities = set(entity2recipes_train.keys())
        missing_entities = [e for e in self.entities if e not in train_entities]

        aux_recipes = set()
        for e in missing_entities:
            aux_recipe = self._random_choice(list(entity2recipes_test[e]))
            aux_recipes.add(aux_recipe)

        for recipe in aux_recipes:
            self.recipes_train.add(recipe)
            self.recipes_test.remove(recipe)

