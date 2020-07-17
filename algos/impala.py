import argparse
import logging
import os
import pprint
import threading
import time
import timeit
import traceback
import typing

os.environ["OMP_NUM_THREADS"] = "1"  # Necessary for multithreading.

import torch
from torch import multiprocessing as mp
from torch import nn
from torch.nn import functional as F

from core import environment
from core import file_writer
from core import prof
from core import vtrace

from utils import seed as utils_seed
from models import RandomAgent, SelfAttentionACNet, SimpleACNet
from graph.kbcr import ComplEx


logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)

Buffers = typing.Dict[str, typing.List[torch.Tensor]]


def model_for_env(flags, env, kg_model=None):
    if flags.env.lower().startswith('wordcraft'):
        if flags.arch == 'random':
            model = RandomAgent(env.observation_space, env.action_space)
        elif 'selfattn' in flags.arch:
            model = SelfAttentionACNet(
                env.observation_space, env.action_space,
                arch=flags.arch,
                use_minmax_features=flags.use_minmax_features,
                kg_model=kg_model,
                key_size=flags.key_size,
                value_size=flags.value_size,
                use_kg_goal_score=flags.use_kg_goal_score,
                use_kg_selection_score=flags.use_kg_selection_score,
                use_kg_only=flags.use_kg_only,
                rb=env.recipe_book)
        elif 'mlp' in flags.arch:
            model = SimpleACNet(
                env.observation_space, env.action_space,
                arch=flags.arch,
                hidden_size=flags.hidden_size)
        else:
            raise ValueError(f'Unsupported arch {arch}.')
    else:
        raise ValueError(f'Unsupported env {env}.')
    return model


def load_kg_model_for_env(flags, env):
    kg_model = None
    if flags.kg_model_path is not None and hasattr(env, 'num_entities'):
        kg_model = ComplEx(env.num_entities, 2, flags.kg_model_embedding_size)
        kg_model_path = os.path.expandvars(os.path.expanduser(flags.kg_model_path))
        kg_model.load_state_dict(torch.load(kg_model_path))
    return kg_model


def compute_baseline_loss(advantages):
    return 0.5 * torch.sum(advantages ** 2)


def compute_entropy_loss(logits):
    """Return the entropy loss, i.e., the negative entropy of the policy."""
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    return torch.sum(policy * log_policy)


def compute_policy_gradient_loss(logits, actions, advantages):
    cross_entropy = F.nll_loss(
        F.log_softmax(torch.flatten(logits, 0, 1), dim=-1),
        target=torch.flatten(actions, 0, 1),
        reduction="none",
    )
    cross_entropy = cross_entropy.view_as(advantages)
    return torch.sum(cross_entropy * advantages.detach())


def test(
    flags,
    env,
    cumulative_steps,
    model: torch.nn.Module,
    split='test',
    logger=None
):
    try:
        logging.info("Tester started.")
        timings = prof.Timings()  # Keep track of how fast things are.
        if flags.seed is None:
            seed = int.from_bytes(os.urandom(4), byteorder="little")
        else:
            seed = flags.seed
        utils_seed(seed)

        env.eval(split=split)

        last_cumulative_steps = -1

        greedy=True

        while True:
            cumulative_steps_now = cumulative_steps.item()
            if last_cumulative_steps < 0 \
                or (cumulative_steps_now - last_cumulative_steps >= flags.test_interval) \
                or cumulative_steps_now >= flags.total_steps:

                logging.info(f"tester: {cumulative_steps.item()} steps")
                env_output = env.initial()
                agent_state = model.initial_state(batch_size=1)
                agent_output, unused_state = model(env_output, agent_state, greedy=greedy)

                episode_count = 0
                mean_episode_return = 0
                while episode_count < flags.num_test_episodes:
                    # Do new rollout.
                    for t in range(flags.unroll_length):
                        timings.reset()

                        with torch.no_grad():
                            agent_output, agent_state = model(env_output, agent_state, greedy=greedy)
                        timings.time("model")

                        env_output = env.step(agent_output["action"])
                        timings.time("step")

                        if env_output["done"][0]:
                            episode_count += 1
                            mean_episode_return += env_output['episode_return'].item()

                mean_episode_return /= flags.num_test_episodes

                last_cumulative_steps = cumulative_steps_now

                to_log = {
                    'step': cumulative_steps_now,
                    'mean_episode_return': mean_episode_return
                }   
                if split == 'test':
                    logger.log_test_test(to_log)
                else:
                    logger.log_test_train(to_log)
                logging.info(f"Tester {split}: %s", timings.summary())

            if cumulative_steps_now >= flags.total_steps:
                logging.info("Tester shutting down.")
                break

    except KeyboardInterrupt:
        pass  # Return silently.
    except Exception as e:
        logging.error("Exception in worker process: tester")
        traceback.print_exc()
        print()
        raise e


def act(
    flags,
    env,
    actor_index: int,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    model: torch.nn.Module,
    buffers: Buffers,
    initial_agent_state_buffers,
):
    try:
        logging.info("Actor %i started.", actor_index)
        timings = prof.Timings()  # Keep track of how fast things are.
        if flags.seed is None:
            seed = actor_index ^ int.from_bytes(os.urandom(4), byteorder="little")
        else:
            seed = flags.seed + actor_index
        utils_seed(seed)

        env_output = env.initial()
        agent_state = model.initial_state(batch_size=1)
        agent_output, unused_state = model(env_output, agent_state)
        while True:
            index = free_queue.get()
            if index is None:
                break

            # Write old rollout end.
            for key in env_output:
                buffers[key][index][0, ...] = env_output[key]
            for key in agent_output:
                buffers[key][index][0, ...] = agent_output[key]
            for i, tensor in enumerate(agent_state):
                initial_agent_state_buffers[index][i][...] = tensor

            # Do new rollout.
            for t in range(flags.unroll_length):
                timings.reset()

                with torch.no_grad():
                    agent_output, agent_state = model(env_output, agent_state)
                timings.time("model")

                env_output = env.step(agent_output["action"])
                timings.time("step")

                for key in env_output:
                    buffers[key][index][t + 1, ...] = env_output[key]
                for key in agent_output:
                    buffers[key][index][t + 1, ...] = agent_output[key]

                timings.time("write")
            full_queue.put(index)

        if actor_index == 0:
            logging.info("Actor %i: %s", actor_index, timings.summary())

    except KeyboardInterrupt:
        pass  # Return silently.
    except Exception as e:
        logging.error("Exception in worker process %i", actor_index)
        traceback.print_exc()
        print()
        raise e


def get_batch(
    flags,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    buffers: Buffers,
    initial_agent_state_buffers,
    timings,
    lock=threading.Lock(),
):
    with lock:
        timings.time("lock")
        indices = [full_queue.get() for _ in range(flags.batch_size)]
        timings.time("dequeue")
    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1) for key in buffers
    }
    initial_agent_state = (
        torch.cat(ts, dim=1)
        for ts in zip(*[initial_agent_state_buffers[m] for m in indices])
    )
    timings.time("batch")
    for m in indices:
        free_queue.put(m)
    timings.time("enqueue")
    batch = {k: t.to(device=flags.device, non_blocking=True) for k, t in batch.items()}
    initial_agent_state = tuple(
        t.to(device=flags.device, non_blocking=True) for t in initial_agent_state
    )
    timings.time("device")
    return batch, initial_agent_state


def learn(
    flags,
    actor_model,
    model,
    batch,
    initial_agent_state,
    optimizer,
    scheduler,
    lock=threading.Lock(),  # noqa: B008
):
    """Performs a learning (optimization) step."""
    with lock:
        learner_outputs, unused_state = model(batch, initial_agent_state)

        # Take final value function slice for bootstrapping.
        bootstrap_value = learner_outputs["baseline"][-1]

        # Move from obs[t] -> action[t] to action[t] -> obs[t].
        batch = {key: tensor[1:] for key, tensor in batch.items()}
        learner_outputs = {key: tensor[:-1] for key, tensor in learner_outputs.items()}

        rewards = batch["reward"]
        if flags.reward_clipping == "abs_one":
            clipped_rewards = torch.clamp(rewards, -1, 1)
        elif flags.reward_clipping == "none":
            clipped_rewards = rewards

        discounts = (~batch["done"]).float() * flags.discounting

        vtrace_returns = vtrace.from_logits(
            behavior_policy_logits=batch["policy_logits"],
            target_policy_logits=learner_outputs["policy_logits"],
            actions=batch["action"],
            discounts=discounts,
            rewards=clipped_rewards,
            values=learner_outputs["baseline"],
            bootstrap_value=bootstrap_value,
        )

        pg_loss = compute_policy_gradient_loss(
            learner_outputs["policy_logits"],
            batch["action"],
            vtrace_returns.pg_advantages,
        )
        baseline_loss = flags.baseline_cost * compute_baseline_loss(
            vtrace_returns.vs - learner_outputs["baseline"]
        )
        entropy_loss = flags.entropy_cost * compute_entropy_loss(
            learner_outputs["policy_logits"]
        )

        total_loss = pg_loss + baseline_loss + entropy_loss

        episode_returns = batch["episode_return"][batch["done"]]
        stats = {
            "episode_returns": tuple(episode_returns.cpu().numpy()),
            "mean_episode_return": torch.mean(episode_returns).item(),
            "total_loss": total_loss.item(),
            "pg_loss": pg_loss.item(),
            "baseline_loss": baseline_loss.item(),
            "entropy_loss": entropy_loss.item(),
        }

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), flags.grad_norm_clipping)
        optimizer.step()
        scheduler.step()

        actor_model.load_state_dict(model.state_dict())
        return stats


def create_buffers(flags, obs_space, num_actions) -> Buffers:
    T = flags.unroll_length
    specs = dict(
        goal_index=dict(size=(T + 1, *obs_space['goal_index'].shape), dtype=torch.int64),
        goal_features=dict(size=(T + 1, *obs_space['goal_features'].shape), dtype=torch.float32),

        table_index=dict(size=(T + 1, *obs_space['table_index'].shape), dtype=torch.int64),
        table_features=dict(size=(T + 1, *obs_space['table_features'].shape), dtype=torch.float32),

        selection_index=dict(size=(T + 1, *obs_space['selection_index'].shape), dtype=torch.int64),
        selection_features=dict(size=(T + 1, *obs_space['selection_features'].shape), dtype=torch.float32),

        reward=dict(size=(T + 1,), dtype=torch.float32),
        done=dict(size=(T + 1,), dtype=torch.bool),
        episode_return=dict(size=(T + 1,), dtype=torch.float32),
        episode_step=dict(size=(T + 1,), dtype=torch.int32),
        policy_logits=dict(size=(T + 1, num_actions), dtype=torch.float32),
        baseline=dict(size=(T + 1,), dtype=torch.float32),
        last_action=dict(size=(T + 1,), dtype=torch.int64),
        action=dict(size=(T + 1,), dtype=torch.int64),
    )

    buffers: Buffers = {key: [] for key in specs}
    for _ in range(flags.num_buffers):
        for key in buffers:
            buffers[key].append(torch.empty(**specs[key]).share_memory_())

    return buffers


def train(flags):  # pylint: disable=too-many-branches, too-many-statements
    if flags.xpid is None:
        flags.xpid = "torchbeast-%s" % time.strftime("%Y%m%d-%H%M%S")
    plogger = file_writer.FileWriter(
        xpid=flags.xpid, xp_args=flags.__dict__, rootdir=flags.savedir
    )
    checkpointpath = os.path.expandvars(
        os.path.expanduser("%s/%s/%s" % (flags.savedir, flags.xpid, "model.tar"))
    )

    if flags.num_buffers is None:  # Set sensible default for num_buffers.
        flags.num_buffers = max(2 * flags.num_actors, flags.batch_size)
    if flags.num_actors >= flags.num_buffers:
        raise ValueError("num_buffers should be larger than num_actors")
    if flags.num_buffers < flags.batch_size:
        raise ValueError("num_buffers should be larger than batch_size")

    T = flags.unroll_length
    B = flags.batch_size

    flags.device = None
    if not flags.disable_cuda and torch.cuda.is_available():
        logging.info("Using CUDA.")
        flags.device = torch.device("cuda")
    else:
        logging.info("Not using CUDA.")
        flags.device = torch.device("cpu")

    gym_env = environment.create_gym_env(flags, seed=flags.seed)

    env = environment.Environment(flags, gym_env)
    env.initial()

    kg_model = load_kg_model_for_env(flags, gym_env)
    model = model_for_env(flags, gym_env, kg_model)

    buffers = create_buffers(flags, gym_env.observation_space, model.num_actions)
    cumulative_steps = torch.zeros(1, dtype=int).share_memory_()

    model.share_memory()

    ctx = mp.get_context("fork")

    tester_processes = []
    if flags.test_interval > 0:
        splits = ['test', 'train']
        for split in splits:        
            tester = ctx.Process(
                target=test,
                args=(
                    flags,
                    env,
                    cumulative_steps,
                    model,
                    split,
                    plogger
                ),
            )
            tester_processes.append(tester)
            tester.start()

    # Add initial RNN state.
    initial_agent_state_buffers = []
    for _ in range(flags.num_buffers):
        state = model.initial_state(batch_size=1)
        for t in state:
            t.share_memory_()
        initial_agent_state_buffers.append(state)

    actor_processes = []
    free_queue = ctx.SimpleQueue()
    full_queue = ctx.SimpleQueue()

    for i in range(flags.num_actors):
        actor = ctx.Process(
            target=act,
            args=(
                flags,
                env,
                i,
                free_queue,
                full_queue,
                model,
                buffers,
                initial_agent_state_buffers,
            ),
        )
        actor.start()
        actor_processes.append(actor)

    learner_model = model_for_env(flags, gym_env, kg_model).to(device=flags.device)

    optimizer = torch.optim.RMSprop(
        learner_model.parameters(),
        lr=flags.learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon,
        alpha=flags.alpha,
    )

    def lr_lambda(epoch):
        return 1 - min(epoch * T * B, flags.total_steps) / flags.total_steps

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    logger = logging.getLogger("logfile")
    stat_keys = [
        "total_loss",
        "mean_episode_return",
        "pg_loss",
        "baseline_loss",
        "entropy_loss",
    ]
    logger.info("# Step\t%s", "\t".join(stat_keys))

    step, stats = 0, {}

    def batch_and_learn(i, lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal step, stats, cumulative_steps
        timings = prof.Timings()
        while step < flags.total_steps:
            timings.reset()
            batch, agent_state = get_batch(
                flags,
                free_queue,
                full_queue,
                buffers,
                initial_agent_state_buffers,
                timings,
            )
            stats = learn(
                flags, model, learner_model, batch, agent_state, optimizer, scheduler
            )
            timings.time("learn")
            with lock:
                to_log = dict(step=step)
                to_log.update({k: stats[k] for k in stat_keys})
                plogger.log(to_log)
                step += T * B
                cumulative_steps[0] = step

        if i == 0:
            logging.info("Batch and learn: %s", timings.summary())

    for m in range(flags.num_buffers):
        free_queue.put(m)

    threads = []
    for i in range(flags.num_learner_threads):
        thread = threading.Thread(
            target=batch_and_learn, name="batch-and-learn-%d" % i, args=(i,)
        )
        thread.start()
        threads.append(thread)

    def checkpoint():
        if flags.disable_checkpoint:
            return
        logging.info("Saving checkpoint to %s", checkpointpath)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "flags": vars(flags),
            },
            checkpointpath,
        )

    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer()
        while step < flags.total_steps:
            start_step = step
            start_time = timer()
            time.sleep(5)

            if timer() - last_checkpoint_time > 10 * 60:  # Save every 10 min.
                checkpoint()
                last_checkpoint_time = timer()

            sps = (step - start_step) / (timer() - start_time)
            if stats.get("episode_returns", None):
                mean_return = (
                    "Return per episode: %.1f. " % stats["mean_episode_return"]
                )
            else:
                mean_return = ""
            total_loss = stats.get("total_loss", float("inf"))
            logging.info(
                "Steps %i @ %.1f SPS. Loss %f. %sStats:\n%s",
                step,
                sps,
                total_loss,
                mean_return,
                pprint.pformat(stats),
            )
    except KeyboardInterrupt:
        return  # Try joining actors then quit.
    else:
        for thread in threads:
            thread.join()
        logging.info("Learning finished after %d steps.", step)
    finally:
        for _ in range(flags.num_actors):
            free_queue.put(None)
        for actor in actor_processes:
            actor.join(timeout=1)
        for tester in tester_processes:
            tester.join(timeout=1)

    checkpoint()
    plogger.close()
