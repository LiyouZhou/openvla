"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment.

Usage:
    # OpenVLA:
    # IMPORTANT: Set `center_crop=True` if model is fine-tuned with augmentations
    python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \
        --center_crop [ True | False ] \
        --run_id_note <OPTIONAL TAG TO INSERT INTO RUN ID FOR LOGGING> \
        --use_wandb [ True | False ] \
        --wandb_project <PROJECT> \
        --wandb_entity <ENTITY>
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import math, torch

import draccus
import numpy as np
import tqdm

# from libero.libero import benchmark

import wandb
import mikasa_robo_suite
import gymnasium as gym
from mikasa_robo_suite.utils.wrappers import StateOnlyTensorToDictWrapper
from mikasa_robo_suite.dataset_collectors.get_mikasa_robo_datasets import env_info

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
# from experiments.robot.libero.libero_utils import (
#     get_libero_dummy_action,
#     get_libero_env,
#     get_libero_image,
#     quat2axisangle,
#     save_rollout_video,
# )
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE,
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
import imageio


def save_rollout_video(rollout_images, idx, success, task_description, log_file=None):
    """Saves an MP4 replay of an episode."""
    rollout_dir = f"./rollouts/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")

    for i, img in enumerate(rollout_images):
        imageio.imwrite(
            f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}--frame={i}.png",
            img,
        )

    return mp4_path


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "mikasa"                  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 0                          # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)

    # [OpenVLA] Set action un-normalization key
    unnorm_key: str = "mikasa_robo_baseline_tfds"
    # fmt: on


@draccus.wrap()
def eval_mikasa(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Load model
    model = get_model(cfg)

    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family == "openvla":
        # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
        # with the suffix "_no_noops" in the dataset name)
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"

        print(model.norm_stats.keys())
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    # Initialize local logging
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    # Initialize LIBERO task suite
    # benchmark_dict = benchmark.get_benchmark_dict()
    # task_suite = benchmark_dict[cfg.task_suite_name]()
    # num_tasks_in_suite = task_suite.n_tasks
    # print(f"Task suite: {cfg.task_suite_name}")
    # log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    num_tasks_in_suite = 1

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # # Get task
        # task = task_suite.get_task(task_id)

        # # Get default LIBERO initial states
        # initial_states = task_suite.get_task_init_states(task_id)

        # # Initialize LIBERO environment and task description
        # env, task_description = get_libero_env(task, cfg.model_family, resolution=256)
        env_name = "RememberColor3-v0"
        # seed = 42
        num_envs = 1
        env_kwargs_rgb = dict(
            num_envs=num_envs,
            obs_mode="rgb",
            control_mode="pd_ee_delta_pose",
            render_mode="all",
            sim_backend="gpu",
            reward_mode="normalized_dense",
            max_episode_steps=100,
        )

        env = gym.make(env_name, **env_kwargs_rgb)
        state_wrappers_list, episode_timeout = env_info(env_name)
        print(f"Episode timeout: {episode_timeout}")
        for wrapper_class, wrapper_kwargs in state_wrappers_list:
            env = wrapper_class(env, **wrapper_kwargs)

        # Start episodes
        task_episodes, task_successes = 0, 0
        dist_to_target = []
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            # Reset environment
            obs, info = env.reset()

            oracle_info = int(info["oracle_info"].cpu())
            color = ["red", "green", "blue"][oracle_info]
            task_description = f"Touch the {color} cube"

            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            # Set initial states
            # obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            if cfg.task_suite_name == "libero_spatial":
                max_steps = 220  # longest training demo has 193 steps
            elif cfg.task_suite_name == "libero_object":
                max_steps = 280  # longest training demo has 254 steps
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 300  # longest training demo has 270 steps
            elif cfg.task_suite_name == "libero_10":
                max_steps = 520  # longest training demo has 505 steps
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400  # longest training demo has 373 steps
            else:
                max_steps = 500  # default max steps

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            terminated = False
            while t < max_steps + cfg.num_steps_wait:
                # try:
                # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                # and we need to wait for them to fall
                if t < cfg.num_steps_wait:
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)
                    t += 1
                    continue

                # Get preprocessed image
                # img = get_libero_image(obs, resize_size)
                img = obs["sensor_data"]["base_camera"]["rgb"][0]
                img = img.cpu().numpy()

                # Save preprocessed image for replay video
                replay_images.append(img)

                # Prepare observations dict
                # Note: OpenVLA does not take proprio state as input
                observation = {
                    "full_image": img,
                    # "state": np.concatenate(
                    #     (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                    # ),
                }

                # Query model to get action
                action = get_action(
                    cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                )
                # print(f"Action: {action}")

                # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
                action = normalize_gripper_action(action, binarize=True)

                # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
                # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
                if cfg.model_family == "openvla":
                    action = invert_gripper_action(action)

                # Execute action in environment
                # print(f"Action: {type(action)} {action.shape} {action}")
                # action = np.append(action, action[-1])
                action = torch.from_numpy(action)
                action = action * 10
                action = torch.stack([action] * num_envs)
                obs, reward, terminated, truncated, info = env.step(action)
                terminated = True if terminated[0].cpu() else False
                truncated = True if truncated[0].cpu() else False
                # print(reward, info)

                if info["success"] == 1:
                    task_successes += 1
                    total_successes += 1
                    break

                if terminated or truncated:
                    break
                t += 1

                # except Exception as e:
                #     print(f"Caught exception: {e}")
                #     log_file.write(f"Caught exception: {e}\n")
                #     break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            mp4_path = save_rollout_video(
                replay_images, total_episodes, success=terminated, task_description=task_description, log_file=log_file
            )

            dist_to_target.append(info["reward_dict"]["tcp_to_obj_dist"].cpu().numpy())
            if cfg.use_wandb:
                wandb.log(
                    {
                        f"rollout_video/{task_description}": wandb.Video(mp4_path, format="mp4"),
                        f"sucess": info["success"].cpu(),
                        f"distance_to_target": info["reward_dict"]["tcp_to_obj_dist"],
                        f"episode_idx": episode_idx,
                    }
                )

            # Log current results
            print(f"Success: {terminated}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {terminated}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

        # Log final results
        avg_dist_to_target = np.mean(dist_to_target)
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        print(f"Average Distance to target: {avg_dist_to_target}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.write(f"Average Distance to target: {avg_dist_to_target}\n")
        log_file.flush()
        if cfg.use_wandb:
            wandb.log(
                {
                    f"{task_description}/success_rate": float(task_successes) / float(task_episodes),
                    f"{task_description}/num_episodes": task_episodes,
                    f"{task_description}/average_distance": avg_dist_to_target,
                }
            )

    # Save local log file
    log_file.close()

    # Push total metrics and local log file to wandb
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": float(total_successes) / float(total_episodes),
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)


if __name__ == "__main__":
    eval_mikasa()
