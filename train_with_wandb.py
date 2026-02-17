"""Custom training script with wandb logging using Stable-Baselines3 SAC."""

import argparse
import sys
import os
from datetime import datetime

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Train an RL agent with Stable-Baselines3 SAC and wandb logging."
)
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--video_interval",
    type=int,
    default=2000,
    help="Interval between video recordings (in steps).",
)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--seed", type=int, default=42, help="Seed used for the environment"
)
parser.add_argument(
    "--total_timesteps", type=int, default=100000, help="Total training timesteps."
)
parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate.")
parser.add_argument(
    "--buffer_size", type=int, default=1000000, help="Replay buffer size."
)
parser.add_argument("--batch_size", type=int, default=256, help="Batch size.")

# Wandb arguments
parser.add_argument(
    "--use_wandb", action="store_true", default=False, help="Enable wandb logging."
)
parser.add_argument(
    "--wandb_project", type=str, default="isaaclab-training", help="Wandb project name."
)
parser.add_argument(
    "--wandb_entity", type=str, default=None, help="Wandb entity/username."
)
parser.add_argument("--wandb_run_name", type=str, default=None, help="Wandb run name.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import numpy as np
from datetime import datetime

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import Stable-Baselines3
try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv

    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print(
        "[WARNING] stable-baselines3 not installed. Install with: pip install stable-baselines3"
    )

# Import wandb
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[WARNING] wandb not installed. Install with: pip install wandb")


class WandbCallback(BaseCallback):
    """Custom callback for logging to wandb."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.step_count = 0

    def _on_step(self) -> bool:
        # Log every 10 steps to avoid too much data
        if self.step_count % 10 == 0:
            logs = self.locals.get("infos", [{}])[0]
            if logs:
                wandb.log(
                    {
                        "train/step": self.num_timesteps,
                        "train/reward": self.locals.get("rewards", [0])[0],
                        "train/episode_length": logs.get("episode", {}).get("r", 0),
                    },
                    step=self.num_timesteps,
                )
        self.step_count += 1
        return True


class WandbLogger:
    """Simple wandb logger for training."""

    def __init__(self, project, entity=None, name=None, config=None):
        if not WANDB_AVAILABLE:
            raise ImportError("wandb not installed")

        self.run = wandb.init(
            project=project, entity=entity, name=name, config=config, reinit=True
        )
        print(f"[INFO] Wandb logging enabled: {self.run.url}")

    def log(self, metrics, step=None):
        """Log metrics to wandb."""
        if self.run:
            wandb.log(metrics, step=step)

    def finish(self):
        """Finish wandb run."""
        if self.run:
            wandb.finish()


def make_env(env_cfg, task_name, seed=0, render_mode=None):
    """Create environment function for SB3."""

    def _init():
        env = gym.make(task_name, cfg=env_cfg, render_mode=render_mode)
        env.reset(seed=seed)
        return env

    return _init


@hydra_task_config(args_cli.task, "sb3_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg):
    """Train with Stable-Baselines3 SAC agent and wandb logging."""

    if not SB3_AVAILABLE:
        print("[ERROR] stable-baselines3 is required for SAC training")
        return

    # Set the environment seed
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )
    env_cfg.scene.num_envs = args_cli.num_envs

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "sb3_sac", args_cli.task.replace("-", "_"))
    log_root_path = os.path.abspath(log_root_path)
    os.makedirs(log_root_path, exist_ok=True)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    # specify directory for logging runs
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args_cli.wandb_run_name:
        log_dir = f"{args_cli.wandb_run_name}_{log_dir}"
    log_dir = os.path.join(log_root_path, log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # Initialize wandb if enabled
    wandb_logger = None
    callbacks = []
    if args_cli.use_wandb and WANDB_AVAILABLE:
        run_name = (
            args_cli.wandb_run_name
            or f"{args_cli.task}_SAC_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        wandb_config = {
            "task": args_cli.task,
            "num_envs": args_cli.num_envs,
            "seed": args_cli.seed,
            "total_timesteps": args_cli.total_timesteps,
            "learning_rate": args_cli.learning_rate,
            "buffer_size": args_cli.buffer_size,
            "batch_size": args_cli.batch_size,
            "algorithm": "SAC",
            "library": "stable-baselines3",
        }
        wandb_logger = WandbLogger(
            project=args_cli.wandb_project,
            entity=args_cli.wandb_entity,
            name=run_name,
            config=wandb_config,
        )
        callbacks.append(WandbCallback())

    # create isaac environment
    render_mode = "rgb_array" if args_cli.video else None
    env = make_env(env_cfg, args_cli.task, args_cli.seed, render_mode)()

    # convert to single-agent instance if required
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Create vectorized environment for SB3
    vec_env = DummyVecEnv([lambda: env])

    # Create SAC model
    print(f"[INFO] Creating SAC model with lr={args_cli.learning_rate}")
    model = SAC(
        "MlpPolicy",
        vec_env,
        learning_rate=args_cli.learning_rate,
        buffer_size=args_cli.buffer_size,
        batch_size=args_cli.batch_size,
        verbose=1,
        tensorboard_log=log_dir if not args_cli.use_wandb else None,
        device="auto",
    )

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)

    # Train the model
    print(f"[INFO] Starting SAC training for {args_cli.total_timesteps} timesteps...")
    model.learn(
        total_timesteps=args_cli.total_timesteps,
        callback=callbacks if callbacks else None,
        progress_bar=True,
    )

    # Save the final model
    model_path = os.path.join(log_dir, "sac_model")
    model.save(model_path)
    print(f"[INFO] Model saved to {model_path}")

    # Log final model to wandb
    if wandb_logger:
        wandb.save(f"{model_path}.zip")
        wandb_logger.finish()

    print("[INFO] Training completed!")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
