# pi05-finetuned-pick-place

## Overview

This repository contains the source code to:
- Define a custom MuJoCo-based manipulation environment using Robosuite
- Record LeRobot datasets in the environment
- Fine tune π0.5, a Vision-Language-Action model by Physical Intelligence
- Run inference

The goal of this endeavour was to demonstrate end-to-end deployment of π0.5. Hence, the environment used is relatively simple: consisting of a blue cube, a yellow cube and a red circle on a checkered table. Work in progress to extend this setup to more advanced manipulation tasks.

A demo can be viewed below.

## Detailed Look

In `organizing_env.py`, a class, `DeskOrganizerRobosuiteEnv`, is defined for the Robosuite environment, which contains functions for spawning objects and randomizing their positions. A Gymnasium wrapper, `DeskOrganizerEnv`, is also defined and registered, allowing for dataset collection and teleoperation with LeRobot's `gym_manipulator.py`. To make recording episodes easier, a simple path planner automatically accesses the coordinates of objects in the scene and guides the manipulator to complete the task. Alternatively, a gamepad can be used for teleoperation. This setting can be changed by toggling the `control_mode` argument between `auto` and `manual` in the constructor for the Gymnasium wrapper. It is set to `auto` by default.
