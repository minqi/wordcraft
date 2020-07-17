import logging
import argparse

from algos import impala


def get_cmd_args():
    parser = argparse.ArgumentParser(description="PyTorch Scalable Agent")

    # Meta settings.
    parser.add_argument("--mode", default="train",
                        choices=["train", "test", "test_render"],
                        help="Training or test mode.")
    parser.add_argument("--xpid", default=None,
                        help="Experiment id (default: None).")
    parser.add_argument("--model_name", type=str, default="test", 
                        help='Name of model')
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed.")

    # Training settings.
    parser.add_argument("--disable_checkpoint", action="store_true",
                        help="Disable saving checkpoint.")
    parser.add_argument("--savedir", default="~/logs/torchbeast",
                        help="Root dir where experiment data will be saved.")
    parser.add_argument("--num_actors", default=4, type=int, metavar="N",
                        help="Number of actors (default: 4).")
    parser.add_argument("--total_steps", default=50000000, type=int, metavar="T",
                        help="Total environment steps to train for.")
    parser.add_argument("--batch_size", default=8, type=int, metavar="B",
                        help="Learner batch size.")
    parser.add_argument("--unroll_length", default=80, type=int, metavar="T",
                        help="The unroll length (time dimension).")
    parser.add_argument("--num_buffers", default=None, type=int,
                        metavar="N", help="Number of shared-memory buffers.")
    parser.add_argument("--num_learner_threads", "--num_threads", default=2, type=int,
                        metavar="N", help="Number learner threads.")
    parser.add_argument("--test_interval", default=5000, type=int,
                        help="Test model every this many steps.")
    parser.add_argument("--num_test_episodes", default=100, type=int,
                        help="Number of episodes used each cycle of testing.")
    parser.add_argument("--disable_cuda", action="store_true",
                        help="Disable CUDA.")

    # Model settings
    parser.add_argument("--arch", type=str, default="selfattn", choices=['random', 'selfattn', 'relational_selfattn', 'mlp', 'mlp_embed'],
                        help="Type of agent.")
    parser.add_argument("--use_minmax_features", action="store_true",
                        help="Combine input vectors by concatenating min and max vectors across inputs")
    parser.add_argument("--key_size", type=int, default=300,
                    help="Key/query dim for self-attention")
    parser.add_argument("--value_size", type=int, default=300,
                    help="Value dim for self-attention")
    parser.add_argument("--hidden_size", type=int, default=300,
                        help="Hidden size of MLP model")

    parser.add_argument("--kg_model_path", type=str, default=None,
                        help="Type of feature extraction architecture")
    parser.add_argument("--use_kg_only", action="store_true",
                        help="Only use KG scores as policy.")
    parser.add_argument("--kg_model_embedding_size", type=int, default=128,
                        help="Embedding size of KBC model.")
    parser.add_argument("--use_kg_goal_score", action="store_true",
                        help="Whether to modulate policy with goal score from KBC model.")
    parser.add_argument("--use_kg_selection_score", action="store_true",
                        help="Whether to modulate policy logits with selection score from KBC model.")

    # Env settings
    parser.add_argument("--env", type=str, default="wordcraft-multistep-goal-v0",
                        help="Environment.")
    parser.add_argument("--data_path", type=str, default="datasets/alchemy2.json",
                        help="Path to data")
    parser.add_argument("--recipe_book_path", type=str, default=None,
                        help="Path to serialized recipe book for loading existing train/test split.")
    parser.add_argument("--split", type=str, default="debug", 
                        choices=['debug', 'by_goal', 'by_recipe', 'by_task', 'by_goal_train_terminals', 'by_recipe_train_all_goals'],
                        help="How tasks are split into train and test sets.")
    parser.add_argument("--train_ratio", type=float, default=1.0,
                        help="Size ratio of train set to test set.")
    parser.add_argument("--feature_type", type=str, default='glove', choices=['one_hot', 'random', 'glove'],
                        help="How entities are represented.")
    parser.add_argument("--shuffle_features", action="store_true",
                        help="Whether to shuffle word embedding features.")
    parser.add_argument("--random_feature_size", type=int, default=300,
                        help="Random feature size.")
    parser.add_argument("--depths", nargs='+', type=int, default=[1],
                        help="Max number of reaction steps to create goal.")
    parser.add_argument("--max_mix_steps", type=int,
                        help="Max number of reaction attempts allowed.")
    parser.add_argument("--num_distractors", type=int, default=1,
                        help="How many distractors to include per task.")
    parser.add_argument("--uniform_distractors", action="store_true",
                        help="How many distractors to include per task.")
    parser.add_argument("--subgoal_rewards", action="store_true",
                        help="Provide subgoal rewards during training.")


    # Loss settings.
    parser.add_argument("--entropy_cost", default=0.001,
                        type=float, help="Entropy cost/multiplier.")
    parser.add_argument("--baseline_cost", default=0.5,
                        type=float, help="Baseline cost/multiplier.")
    parser.add_argument("--discounting", default=0.99,
                        type=float, help="Discounting factor.")
    parser.add_argument("--reward_clipping", default="none",
                        choices=["abs_one", "none"],
                        help="Reward clipping.")

    # Optimizer settings.
    parser.add_argument("--learning_rate", default=0.0005,
                        type=float, metavar="LR", help="Learning rate.")
    parser.add_argument("--alpha", default=0.99, type=float,
                        help="RMSProp smoothing constant.")
    parser.add_argument("--momentum", default=0, type=float,
                        help="RMSProp momentum.")
    parser.add_argument("--epsilon", default=1e-8, type=float,
                        help="RMSProp epsilon.")
    parser.add_argument("--grad_norm_clipping", default=40.0, type=float,
                        help="Global gradient norm clip.")

    # Log settings.
    parser.add_argument("--verbose", action="store_true",
                        help="Whether to print logs")

    return parser.parse_args()


if __name__ == '__main__':
    flags = get_cmd_args()

    if not flags.verbose:
        logging.disable(logging.CRITICAL)

    impala.train(flags)

