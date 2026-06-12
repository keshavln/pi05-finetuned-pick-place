import os
import torch
import shutil
from tqdm import tqdm
from lerobot.datasets import LeRobotDataset

def main():
    old_repo_id = "KeshavLN/deskorgv1.1"
    new_repo_id = "KeshavLN/deskorgv1.2_dataset"
    
    old_dataset = LeRobotDataset(old_repo_id)
        
    # Define the new 8D state and 7D action features
    new_features = old_dataset.features.copy()
    new_features["observation.state"] = {
        "dtype": "float32",
        "shape": (8,),
        "names": ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"]
    }
    new_features["action"] = {
        "dtype": "float32",
        "shape": (7,),
        "names": ["dx", "dy", "dz", "drx", "dry", "drz", "dgripper"]
    }
    
    new_dataset = LeRobotDataset.create(
        repo_id=new_repo_id,
        fps=old_dataset.fps,
        robot_type="panda",
        features=new_features,
        use_videos=True,
    )
    
    episode_indices = old_dataset.hf_dataset["episode_index"]
    
    current_ep = -1
    task = ""
    
    for i in tqdm(range(len(old_dataset)), desc="Processing Frames"):
        ep_idx = episode_indices[i]
        
        if ep_idx != current_ep:
            if current_ep != -1:
                new_dataset.save_episode()
            current_ep = ep_idx
            
            task = "organize the desk"
            try:
                task_idx = old_dataset.hf_dataset["task_index"][i]
                if hasattr(task_idx, "item"):
                    task_idx = task_idx.item()
                
                if hasattr(old_dataset, 'meta') and hasattr(old_dataset.meta, 'tasks'):
                    tasks_df = old_dataset.meta.tasks
                    # Handle LeRobot v3 Pandas DataFrame format
                    if hasattr(tasks_df, 'index'):
                        task_str = tasks_df[tasks_df['task_index'] == task_idx].index[0]
                        task = str(task_str)
                    elif isinstance(tasks_df, dict):
                        # Dictionary mapping string to integer (older versions)
                        for t_str, t_idx in tasks_df.items():
                            if t_idx == task_idx:
                                task = str(t_str)
                                break
            except Exception as e:
                print(f"Warning: Could not extract task, defaulting to 'organize the desk'. Error: {e}")
                
        frame = old_dataset[i]
        
        # 1. Slice state from 9D to 8D (x, y, z, qx, qy, qz, qw, gripper)
        # Drop the second gripper joint to match Libero
        state = frame["observation.state"]
        # Normalize gripper state to [-1, 1] where applicable (assuming raw is 0.0 or 2.0)
        grip_state = state[7].item() - 1.0 if state[7].item() > 0.5 else -1.0
        
        state_8d = torch.tensor([
            state[0], state[1], state[2],
            state[3], state[4], state[5], state[6],
            grip_state
        ], dtype=state.dtype)
        
        # 2. Pad action from 4D to 7D (dx, dy, dz, 0, 0, 0, gripper)
        # Normalize gripper action to [-1, 1] mapped from [0.0, 2.0]
        action = frame["action"]
        grip_action = action[3].item() - 1.0
        
        action_7d = torch.tensor([
            action[0], action[1], action[2],
            0.0, 0.0, 0.0,                   
            grip_action                     
        ], dtype=action.dtype)
        
        new_frame = {
            "observation.state": state_8d,
            "action": action_7d,
            "observation.images.image": frame["observation.images.image"],
            "observation.images.image2": frame["observation.images.image2"],
            "task": task,
        }
        
        # Copy any remaining features (like next.reward, next.done, etc.)
        reserved_keys = {"index", "timestamp", "frame_index", "episode_index", "task_index", "task"}
        for key in new_features:
            if key in reserved_keys:
                continue
            if key not in new_frame:
                val = frame[key] if key in frame else old_dataset.hf_dataset[key][i]
                
                # Fix shape mismatch
                if isinstance(val, (int, float, bool)):
                    val = torch.tensor([val], dtype=torch.float32 if isinstance(val, float) else torch.int64)
                elif isinstance(val, torch.Tensor) and val.ndim == 0:
                    val = val.view(1)
                    
                new_frame[key] = val
        
        new_dataset.add_frame(new_frame)
        
    if current_ep != -1:
        new_dataset.save_episode()
        
    new_dataset.push_to_hub()

if __name__ == "__main__":
    main()
