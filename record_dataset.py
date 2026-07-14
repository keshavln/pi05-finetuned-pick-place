"""
Standalone dataset recorder for DeskOrganizerEnv. 
Records demonstrations in the LeRobotDataset v3.0 format. 
The input source (gamepad/path planner) can be set using the control_mode parameter when instantiating DeskOrganizerEnv. 

"""

import time
import logging
import numpy as np
import torch
from PIL import Image

import os
os.environ["MUJOCO_GL"] = "glx"

import gymnasium
gymnasium.envs.registration.register(
    id="gym_hil/DeskOrganizer-v0",
    entry_point="organizing_env:DeskOrganizerEnv",
)

from organizing_env import DeskOrganizerEnv
from lerobot.datasets import LeRobotDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def make_features() -> dict:
    """Define the LeRobot feature schema for our dataset."""
    return {
        # 7-DoF EE delta action: [dx, dy, dz, droll, dpitch, dyaw, gripper]
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"],
        },
        # Proprioceptive state: [eef_pos(3), eef_quat(4), gripper_qpos(2)] = 9D
        "observation.state": {
            "dtype": "float32",
            "shape": (9,),
            "names": None,
        },
        # Front camera (third-person view)
        "observation.images.image": {
            "dtype": "video",
            "shape": (3, 256, 256),
            "names": ["channels", "height", "width"],
        },
        # Wrist camera (eye-in-hand)
        "observation.images.image2": {
            "dtype": "video",
            "shape": (3, 256, 256),
            "names": ["channels", "height", "width"],
        },
    }

def task_plan():
    # key: task string, value: tuple containing target object, destination object and number of episodes to record.
    # note:
    # - mentioning the target and destination objects is necessary to guarantee that they are spawned on the table.
    # - the target and destination object names are not passed to the model. the only text input is the task string.
    task_dict = {
        "Put the coffee mug on the coaster."   : ('mug_b', 'coaster_a', 11),
        "Put the holder on the coaster."       : ('holder_a', 'coaster_a', 11),
        "Put the coffee mug on the pad."       : ('mug_b', 'pad', 11),
        "Put the pen on the pad."              : ('pen_a', 'pad', 11),
        "Put the highlighter on the pad."      : ('highlighter', 'pad', 11),
        "Put the mouse on the pad."            : ('mouse', 'pad', 11),
        "Put the calculator on the pad."       : ('calculator', 'pad', 11)
    }
    task_list = [task for task in task_dict.keys() for _ in range(task_dict[task][2])]

    return task_dict, task_list

def main():
    fps = 10
    control_time_s = 45.0
    max_steps = int(control_time_s * fps)
    control_mode = "auto"

    repo_id = 'KeshavLN/deskorgv2.3_dataset_nv_4'
    root = None
    push_to_hub = False
    fast_mode = False 

    task_dict, task_list = task_plan()
    num_episodes = len(task_list)

    try:
        print(f"Checking for existing dataset at {repo_id}")
        dataset = LeRobotDataset.resume(
            repo_id=repo_id,
            root=root,
            image_writer_threads=4,
        )
        print("Found existing dataset! Appending to it.")
    except Exception:
        print("Dataset not found. Creating a new one.")
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            root=root,
            features=make_features(),
            use_videos=True,
            image_writer_threads=4,
        )

    dt = 1.0 / fps

    print(f"Recording {num_episodes} episodes at {fps} FPS ({control_time_s}s / episode)")
    print(f"Control mode: {control_mode}")
    print(f"Dataset: {repo_id}")

    episode_idx = 0

    try:
        while episode_idx < num_episodes:
            episode_task = task_list[episode_idx]
            # re-initialize the environment for every episode to ensure full domain randomization
            env = DeskOrganizerEnv(
                has_renderer=True,
                control_mode=control_mode,
                has_offscreen_renderer=True,
                image_obs=True,
                control_freq=20,
                horizon=1000,
                target_object=task_dict[episode_task][0],
                destination_object=task_dict[episode_task][1],
                make_vertical=False
            )
            obs, info = env.reset()
            
            rerecord_flag = [False]
            def _key_cb(keycode):
                if keycode in (ord('L'), ord('l')):
                    rerecord_flag[0] = True
            
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(
                env._env.sim.model._model, env._env.sim.data._data,
                key_callback=_key_cb
            )
            viewer.opt.geomgroup[0] = 0
            viewer.opt.geomgroup[1] = 1
            
            viewer.cam.azimuth = 140
            viewer.cam.elevation = -40
            viewer.cam.distance = 1.0
            viewer.cam.lookat[:] = [0.1, 0.0, 0.85]
            step_count = 0
            episode_start = time.perf_counter()

            print(f"── Episode {episode_idx + 1}/{num_episodes} ──")

            while step_count < max_steps:
                t_start = time.perf_counter()

                # send neutral action, the env internally overrides with
                # gamepad input or path planner output when active
                neutral = np.zeros(7, dtype=np.float32)
                obs, reward, terminated, truncated, info = env.step(neutral)
                
                if viewer.is_running():
                    if not fast_mode or step_count % 20 == 0:
                        viewer.sync()
                
                if rerecord_flag[0]:
                    info["rerecord_episode"] = True
                    break

                executed_action = info.get("executed_action", neutral)

                frame = {
                    "action": torch.from_numpy(executed_action).float(),
                    "observation.state": torch.from_numpy(obs["agent_pos"]).float(),
                    "observation.images.image": Image.fromarray(obs["pixels"]["image"]),
                    "observation.images.image2": Image.fromarray(obs["pixels"]["image2"]),
                    "task": episode_task,
                }
                dataset.add_frame(frame)
                step_count += 1

                if terminated or truncated:
                    break

                elapsed = time.perf_counter() - t_start
                if fast_mode:
                    time.sleep(max((dt / 20.0) - elapsed, 0))
                else:
                    time.sleep(max(dt - elapsed, 0))

            # ── Episode finished ──
            episode_time = time.perf_counter() - episode_start

            if info.get("rerecord_episode", False):
                print(f"Re-recording episode (discarding {step_count} steps)")
                dataset.clear_episode_buffer()
            else:
                print(f"Saved episode {episode_idx + 1} "
                            f"({step_count} steps, {episode_time:.1f}s)")
                dataset.save_episode()
                episode_idx += 1
                
            if viewer.is_running():
                viewer.close()
            env.close()

    except KeyboardInterrupt:
        print("\nRecording interrupted by user.")
        if dataset.has_pending_frames():
            print("Discarding incomplete episode buffer.")
            dataset.clear_episode_buffer()

    finally:
        print("Finalizing dataset...")
        dataset.finalize()

        if push_to_hub and episode_idx > 0:
            print("Pushing dataset to Hugging Face Hub...")
            dataset.push_to_hub()
            print("Push complete!")

        if 'env' in locals() and hasattr(env, 'close'):
            env.close()

        print(f"Done. Recorded {episode_idx} episode(s) to {repo_id}.")


if __name__ == "__main__":
    main()
