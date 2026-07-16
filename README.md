# Fine Tuning the π0.5 VLA for a Complex Desk Environment

## Overview

This repository contains the source code to:
- Create and randomize a complex MuJoCo-based manipulation environment using Robosuite
- Record LeRobot datasets in the environment
- Fine tune π0.5, a Vision-Language-Action model by Physical Intelligence
- Run inference on various manipulation tasks

The goal of this endeavour was to demonstrate end-to-end deployment of π0.5 and adapt it to perform tasks in various lighting and background conditions. The environment consists of an assortment of common desk items randomly placed across a table. These items include pens, highlighters, mugs, coasters, etc. Domain randomization includes randomizing table colour, background hdri, lightning conditions and shadows. A custom LeRobot dataset of more than 600 episodes was recorded in this environment and used to fine tune π0.5. The training process used LoRA with a rank of 64 over 70,000 steps. 

The result is a VLA that is able to interact with objects of various dimensions across a range of environmental conditions. The only inputs to the model are front view and wrist cam RGB images, proprioceptive state and natural language prompt. No depth data is used. The fine-tuned policy can be found [here](https://huggingface.co/KeshavLN/deskorgv2.7_policy), and the dataset used can be found [here](https://huggingface.co/datasets/KeshavLN/deskorgv2.4_dataset_nv).

A demo can be viewed below.




https://github.com/user-attachments/assets/6724b13c-d9e8-4b8b-a132-2722a31f98ab



## Detailed Look

In `organizing_env.py`, a class, `DeskOrganizerRobosuiteEnv`, is defined for the Robosuite environment, which contains functions for spawning objects and randomizing their positions. A Gymnasium wrapper, `DeskOrganizerEnv`, is also defined and registered. The [obj2mjcf](https://github.com/kevinzakka/obj2mjcf) CLI was used to load textured objects into MuJoCo and generate accurate collision meshes.

While the first iteration of this project used LeRobot's built-in `gym_manipulator.py` for teleoperation and data collection, the pipeline now uses a custom recording script, `record_dataset.py`. To make recording episodes easier, a simple path planner automatically accesses the coordinates of objects in the scene and guides the manipulator to complete the task. Note that this functionality is solely used for recording expert trajectories used to train the model. Location information of objects in the scene is not used by the VLA during inference or training. A single episode takes roughly 30 seconds to record autonomously. 

Alternatively, a gamepad can be used for teleoperation. This setting can be changed by toggling the `control_mode` argument between `auto` and `manual` when instantiating the Gymnasium wrapper, `DeskOrganizerEnv`. It is set to `manual` in the inference script. The tasks to record, number of episodes to record for each task, control time, HuggingFace repository, and additional details are all specified directly in `record_dataset.py`. Keep in mind that terminal authentication is required via `hf auth login` before a locally recorded dataset can be pushed to HuggingFace.

A pre-existing π0.5 checkpoint fine-tuned on the libero environment was chosen for further parameter-efficient fine-tuning. 118 million parameters were trained over a total of 70,000 steps. Relative actions were not used. `inference.ipynb` contains the final inference code, and makes use of libero's processors.

## Usage
```
# Install LeRobot with pi05 dependencies
pip install "lerobot[pi]"

# Clone this repository
git clone https://github.com/keshavln/pi05-finetuned-pick-place.git

# Run inference.ipynb
# Edit the task dict in the second cell to specify the tasks and number of episodes for each.
# At the moment, target and destination object names must also be specified in the task dict solely to guarantee the
# appearance of those objects in the inference environment. This will be altered in future versions of this project.
# For training and other useful commands, refer to commands.txt.

```
