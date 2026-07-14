# Fine Tuning π0.5 in a Custom Robosuite Environment

## Overview

This repository contains the source code to:
- Create and randomize a complex MuJoCo-based manipulation environment using Robosuite
- Record LeRobot datasets in the environment
- Fine tune π0.5, a Vision-Language-Action model by Physical Intelligence
- Run inference on various manipulation tasks

The goal of this endeavour was to demonstrate end-to-end deployment of π0.5 and adapt it to perform tasks in varying lighting and background conditions. The environment consists of an assortment of common desk items randomly placed across a table. These items include pens, highlighters, mugs, coasters, etc. Domain randomization includes randomizing table colour, background hdri, lightning conditions and shadows. A custom LeRobot dataset of more than 600 episodes was recorded in this environment and used to fine tune π0.5. The training process used LoRA with a rank of 64 over 70,000 steps. 

The result is a VLA that is able to interact with objects of various dimensions across a range of environmental conditions. The only inputs to the model are front view and wrist cam RGB images, proprioceptive state and natural language prompt. No depth data is used. The fine-tuned policy can be found [here](https://huggingface.co/KeshavLN/deskorgv2.7_policy), and the dataset used can be found [here](https://huggingface.co/datasets/KeshavLN/deskorgv2.4_dataset_nv).

A demo can be viewed below.


https://github.com/user-attachments/assets/2c6f4985-e277-4f19-90d2-c20d764bf68a


## Detailed Look

In `organizing_env.py`, a class, `DeskOrganizerRobosuiteEnv`, is defined for the Robosuite environment, which contains functions for spawning objects and randomizing their positions. A Gymnasium wrapper, `DeskOrganizerEnv`, is also defined and registered. The [obj2mjcf](https://github.com/kevinzakka/obj2mjcf) CLI was used to load textured objects into MuJoCo and generate accurate collision meshes. While the first iteration of this project used LeRobot's built-in `gym_manipulator.py` for teleoperation and data collection, the pipeline now uses a custom recording script, `record_dataset.py`.

To make recording episodes easier, a simple path planner automatically accesses the coordinates of objects in the scene and guides the manipulator to complete the task. Note that this functionality is solely used for recording expert trajectories. Location information of objects in the scene is not used by the VLA during inference, or training. Alternatively, a gamepad can be used for teleoperation. This setting can be changed by toggling the `control_mode` argument between `auto` and `manual` in the constructor for the Gymnasium wrapper. It is set to `auto` by default. The tasks to record, number of episodes to record for each task, control time, HuggingFace repository, and additional details are all specified directly in `record_dataset.py`. Keep in mind that terminal authentication is required via `hf auth login` before a locally recorded dataset can be pushed to HuggingFace.

A pre-existing π0.5 checkpoint fine-tuned on the libero environment was chosen for further parameter-efficient fine-tuning, and yielded much better results compared to the base model. Relative actions were not used. `inference.ipynb` contains the final inference code, and makes use of libero's processors.
