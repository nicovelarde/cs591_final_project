"""
PPO Training Orchestration Script - RUN ALL AT ONCE
====================================================

This script runs ALL experiments at once (no batching).
Runs all 9 configurations with 1 seed each = 9 total runs.

UPDATED CONFIGURATION (YOUR SPECIFICATIONS):
============================================
VARIED HYPERPARAMETERS:
- Learning Rate: [1e-4, 5e-5, 3e-5] (3 values)
- Entropy Coefficient: [0.0, 0.01, 0.02] (3 values)
- Total configs: 3 × 3 = 9

FIXED HYPERPARAMETERS:
- Clip Ratio: 0.2
- GAE Lambda: 0.95
- Gamma (Discount): 0.99
- Epochs per Update: 5
- Total Episodes: 5000
- Batch Size: 64
- Rollout Steps: 2048

TOTAL RUNS: 27 (9 configs × 3 seeds)
ESTIMATED TIME: ~20 GPU hours (continuous)
"""

import json
import ale_py
import os
import numpy as np
from itertools import product
from datetime import datetime


# ============================================================================
# HYPERPARAMETER CONFIGURATION
# ============================================================================


def create_base_config() -> dict:
    """Create the base configuration dictionary."""
    return {
        "algorithm": "PPO",
        "environment": "ALE/Pong-v5",
        "training": {
            "total_episodes": 5000,  # FIXED
            "rollout_steps": 2048,
            "epochs_per_update": 5,  # FIXED
            "batch_size": 64,
        },
        "network": {
            "hidden_sizes": [512],
            "activation": "relu",
            "shared_extractor": True,
        },
        "hyperparameters": {
            "learning_rate": 3e-5,  # VARIED: [1e-4, 5e-5, 3e-5]
            "gamma": 0.99,  # FIXED
            "gae_lambda": 0.95,  # FIXED
            "clip_ratio": 0.2,  # FIXED
            "entropy_coeff": 0.01,  # VARIED: [0.0, 0.01, 0.02]
            "value_coeff": 0.5,
            "max_grad_norm": 0.5,
        },
        "exploration": {"type": "entropy", "entropy_schedule": "constant"},
    }


def create_experiment_configs() -> list:
    """
    Create all 9 experiment configurations.

    Grid: 3 learning rates × 3 entropy coefficients = 9 configs
    """
    # YOUR SPECIFICATION:
    learning_rates = [1e-4, 5e-5, 3e-5]  # High, Medium, Low
    entropy_coeffs = [0.0, 0.01, 0.02]  # None, Low, Medium

    configs = []
    config_index = 0

    for lr, entropy in product(learning_rates, entropy_coeffs):
        config = create_base_config()
        config["hyperparameters"]["learning_rate"] = lr
        config["hyperparameters"]["entropy_coeff"] = entropy

        config_name = f"ppo_lr{lr:.0e}_entropy{entropy:.2f}".replace("e-0", "e")

        configs.append(
            {
                "index": config_index,
                "name": config_name,
                "config": config,
                "lr": lr,
                "entropy": entropy,
            }
        )
        config_index += 1

    return configs


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================


class ExperimentRunner:
    """Runs all experiments at once."""

    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        self.logs_dir = os.path.join(output_dir, "logs")
        self.configs_dir = os.path.join(output_dir, "configs")

        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.configs_dir, exist_ok=True)

        self.all_results = []

    def save_config(self, config_info: dict) -> str:
        """Save config to JSON file."""
        config_name = config_info["name"]
        config_path = os.path.join(self.configs_dir, f"{config_name}.json")

        with open(config_path, "w") as f:
            json.dump(config_info["config"], f, indent=2)

        return config_path

    def run_single_experiment(self, config_info: dict, seed: int) -> dict:
        """Run one training experiment."""
        import torch
        import gymnasium as gym
        from gymnasium.wrappers import GrayscaleObservation
        from ppo_pong import PPOTrainer

        # Set seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        config = config_info["config"]
        config_name = config_info["name"]

        print(f"  Seed {seed}: ", end="", flush=True)

        # Create environment
        env = gym.make(config["environment"], render_mode=None)
        env = GrayscaleObservation(env)

        # Set seed via reset (gymnasium way) - don't use env.seed() which is deprecated
        # The actual seeding happens in the environment reset

        # Train
        trainer = PPOTrainer(env, config, seed=seed)
        metrics = trainer.train(
            total_episodes=config["training"]["total_episodes"], log_interval=100
        )

        env.close()

        # Save model
        model_path = os.path.join(self.logs_dir, f"{config_name}_seed{seed}_model.pt")
        trainer.save_model(model_path)

        # Save CSV
        import csv

        csv_path = os.path.join(self.logs_dir, f"{config_name}_seed{seed}.csv")

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "reward", "episode_length"])
            for ep, (reward, length) in enumerate(
                zip(metrics["episode_rewards"], metrics["episode_lengths"])
            ):
                writer.writerow([ep, reward, length])

        final_reward = np.mean(metrics["episode_rewards"][-100:])
        print(f"Done! Final: {final_reward:.2f}")

        results = {
            "config_name": config_name,
            "seed": seed,
            "learning_rate": config["hyperparameters"]["learning_rate"],
            "entropy_coeff": config["hyperparameters"]["entropy_coeff"],
            "episode_rewards": metrics["episode_rewards"],
            "episode_lengths": metrics["episode_lengths"],
            "model_path": model_path,
        }

        return results

    def run_all_experiments(self, num_seeds: int = 3):
        """Run all 9 configurations with multiple seeds."""
        configs = create_experiment_configs()

        print("\n" + "=" * 70)
        print("PPO PONG - RUNNING ALL EXPERIMENTS AT ONCE")
        print("=" * 70)
        print(f"\nConfigurations: {len(configs)}")
        print(f"Seeds per config: {num_seeds}")
        print(f"Total runs: {len(configs) * num_seeds} = {len(configs)} × {num_seeds}")
        print(f"\nEstimated time: ~{len(configs) * num_seeds * 0.67:.0f} GPU hours")
        print(f"                (assuming ~40 min per run on modern GPU)")
        print(f"\nWith GPU available, this should take: 13-16 hours continuously")
        print("=" * 70 + "\n")

        start_time = datetime.now()

        for i, config_info in enumerate(configs):
            print(f"\n[{i+1}/{len(configs)}] {config_info['name']}")
            print(
                f"  LR: {config_info['lr']:.0e}, Entropy: {config_info['entropy']:.2f}"
            )

            self.save_config(config_info)
            config_results = []

            for seed in range(num_seeds):
                try:
                    results = self.run_single_experiment(config_info, seed)
                    config_results.append(results)
                    self.all_results.append(results)
                except Exception as e:
                    print(f"    ERROR: {e}")

            # Summary for this config
            if config_results:
                rewards = [np.mean(r["episode_rewards"][-100:]) for r in config_results]
                print(f"\n  Summary: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
                print(
                    f"           (range: {np.min(rewards):.2f} - {np.max(rewards):.2f})"
                )

        elapsed = datetime.now() - start_time
        print(f"\n{'='*70}")
        print(f"COMPLETE! Total time: {elapsed}")
        print(f"{'='*70}\n")

        self.save_summary()
        print("Now run: python utils.py")
        print("This will generate all plots and analysis!")

    def save_summary(self):
        """Save summary of all results."""
        summary_path = os.path.join(self.logs_dir, "experiment_summary.txt")

        with open(summary_path, "w") as f:
            f.write("PPO PONG - ALL EXPERIMENTS SUMMARY\n")
            f.write("=" * 70 + "\n\n")

            f.write("HYPERPARAMETER CONFIGURATION:\n\n")
            f.write("FIXED:\n")
            f.write("  Clip Ratio: 0.2\n")
            f.write("  GAE Lambda: 0.95\n")
            f.write("  Gamma: 0.99\n")
            f.write("  Epochs per Update: 5\n")
            f.write("  Total Episodes: 5000\n\n")

            f.write("VARIED:\n")
            f.write("  Learning Rate: [1e-4, 5e-5, 3e-5]\n")
            f.write("  Entropy Coefficient: [0.0, 0.01, 0.02]\n")
            f.write("  Total Configs: 9\n")
            f.write("  Seeds per Config: 3\n")
            f.write("  Total Runs: 27\n\n")

            f.write("=" * 70 + "\n\n")

            # Group by config
            configs_dict = {}
            for result in self.all_results:
                config_name = result["config_name"]
                if config_name not in configs_dict:
                    configs_dict[config_name] = []
                configs_dict[config_name].append(result)

            # Write results
            for config_name in sorted(configs_dict.keys()):
                results_list = configs_dict[config_name]

                f.write(f"\n{config_name}\n")
                f.write("-" * 70 + "\n")

                final_rewards = [
                    np.mean(r["episode_rewards"][-100:]) for r in results_list
                ]

                f.write(f"LR: {results_list[0]['learning_rate']:.0e}\n")
                f.write(f"Entropy: {results_list[0]['entropy_coeff']:.2f}\n")
                f.write(
                    f"Final Reward: {np.mean(final_rewards):.2f} ± {np.std(final_rewards):.2f}\n"
                )
                f.write(
                    f"Range: [{np.min(final_rewards):.2f}, {np.max(final_rewards):.2f}]\n"
                )


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    runner = ExperimentRunner()
    runner.run_all_experiments(num_seeds=1)
