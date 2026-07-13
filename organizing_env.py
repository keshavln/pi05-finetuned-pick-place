from OpenGL.raw.GL.SGIS.point_line_texgen import GL_OBJECT_DISTANCE_TO_LINE_SGIS
import gymnasium as gym
import numpy as np
import os
import cv2
import time
import mujoco
from random import random, randint, choice
from pathlib import Path
from gymnasium.envs.registration import register
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.tasks import ManipulationTask
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject, CylinderObject, MujocoXMLObject, CerealObject, MilkObject
from robosuite.models.objects import BottleObject, CanObject, PotWithHandlesObject
from robosuite.models.objects import MujocoXMLObject
from robosuite.controllers import load_controller_config
from robosuite.utils.mjcf_utils import new_element
import robosuite.utils.transform_utils as T
from gym_hil.wrappers.intervention_utils import GamepadController

os.environ['MUJOCO_GL'] = 'glx'

CONTROL_FREQ = 20
EE_STEP_SIZE = 0.5

register(
    id='gym_hil/DeskOrganizer-v0',
    entry_point='organizing_env:DeskOrganizerEnv'
)

class DeskOrganizerRobosuiteEnv(SingleArmEnv):
    def __init__(self, **kwargs):
        kwargs.setdefault('task_desc', 'bing chilling')
        self.task_desc = kwargs.pop("task_desc")
        self.target_object = kwargs.pop("target_object")
        self.destination_object = kwargs.pop("destination_object")
        kwargs.setdefault("robots", "Panda")
        kwargs.setdefault("gripper_types", "PandaGripper")
        kwargs.setdefault("has_renderer", False)  
        kwargs.setdefault("has_offscreen_renderer", True)
        kwargs.setdefault("use_camera_obs", True)
        kwargs.setdefault("camera_names", ["frontview", "robot0_eye_in_hand"])
        kwargs.setdefault("camera_heights", 256)
        kwargs.setdefault("camera_widths", 256)
        kwargs.setdefault("render_camera", "frontview")
        kwargs.setdefault("control_freq", CONTROL_FREQ)
        ctrl_config = load_controller_config(default_controller="OSC_POSE")
        # adjusting the kp values of the controller for snappier movement
        ctrl_config["kp"] = [2000, 2000, 2000, 150, 150, 150]
        kwargs.setdefault("controller_configs", ctrl_config)
        self.height=0.9
        import json
        with open('item_registry.json', 'r') as f:
            registry_data = json.load(f)

        CLASS_MAP = {
            "BoxObject": BoxObject,
            "CylinderObject": CylinderObject,
            "BottleObject": BottleObject,
            "CanObject": CanObject,
            "MilkObject": MilkObject,
            "CerealObject": CerealObject,
            "MujocoXMLObject": MujocoXMLObject
        }

        self.item_registry = {}
        for key, val in registry_data.items():
            self.item_registry[key] = {
                "class": CLASS_MAP[val["class"]],
                "kwargs": val["kwargs"]
            }
        

        self.scene_items_input = {
            "can"         : 0,
            "mug_a"       : 0,
            "pen_a"       : 0,
            "pencil"      : 0,
            "scissors"    : 0,
            "highlighter" : 0,
            "calculator"  : 0,
            "holder_a"    : 0,
            "mouse"       : 0,
            "coaster_a"   : 0,
            "mug_b"       : 0,
            "spoon"       : 0,
            "pad"         : 0,
        }
        # choosing objects to spawn at random; target and destination object are guaranteed.
        self.scene_items_input[self.target_object] += 1
        self.scene_items_input[self.destination_object] += 1
        choice_list = list(self.scene_items_input.keys())*2
        choice_list.remove(self.target_object)
        choice_list.remove(self.destination_object)
        for _ in range(5):
            self.scene_items_input[choice(choice_list)] += 1

        self.scene_items = [obj for obj in self.scene_items_input for _ in range(self.scene_items_input[obj])]
        self.mujoco_objects = []
        self.objects_dict = {}
        self.object_ids = {}
        super().__init__(**kwargs)

    def _return_randoms(self, items: list, x_max=0.27, x_min=0.03, y_max=0.32, y_min=-0.32, margin=0.15):
        """
        Returns a set of random coordinates to place objects, separated by a minimum distance.
        """
        coords = {}
        item_ctr = step_ctr = 0
        for item in items:
            too_close = True
            while too_close:
                too_close = False
                new = np.array([np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max)])
                for pair in coords.values():
                    if np.linalg.norm(pair - new) < margin:
                        too_close = True
                step_ctr += 1
                if step_ctr % 50000 == 49999:
                    print(f"taking a long time to converge at {item_ctr}, reducing margin.")
                    margin -= 0.02
            coords[item] = new
            item_ctr += 1
            step_ctr = 0
        return coords
        

    def _load_model(self):
        super()._load_model()
        
        # increase actuator force
        gripper_root = self.robots[0].gripper.root
        for actuator in gripper_root.findall(".//actuator/position"):
            if "gripper_finger_joint" in actuator.get("name", ""):
                actuator.set("forcerange", "-140 140")
                
        # increase only torsional friction on the pads to prevent objects from spinning
        for geom in gripper_root.findall(".//geom"):
            if "pad_collision" in geom.get("name", ""):
                geom.set("friction", "2 1.0 0.0001")
        
        self.table_full_size = (0.5, 1, 0.05)
        self.table_offset = np.array((0.15, 0, self.height))

        # adjusting robot base position
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)
        
        self.mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=(1.0, 0.005, 0.0001),
            table_offset=self.table_offset
        )
        self.mujoco_arena.set_origin([0, 0, 0])

        for geom in self.mujoco_arena.worldbody.findall("./geom"):
            if "wall" in geom.get("name", ""):
                self.mujoco_arena.worldbody.remove(geom)
            elif "floor" in geom.get("name", ""):
                geom.set("rgba", "0 0 0 0")

        
        grid_tex = new_element("texture", name="desk_grid", type="2d", builtin="checker", 
                               rgb1="0.6 0.6 0.6", rgb2="0.5 0.5 0.5", width="512", height="512")
        grid_mat = new_element("material", name="desk_mat", texture="desk_grid", 
                               texrepeat="10 10", texuniform="true", reflectance="0.0")
        
        self.mujoco_arena.asset.append(grid_tex)
        self.mujoco_arena.asset.append(grid_mat)
        
        hdri_directory = Path('assets/hdri')
        hdris = [image for image in hdri_directory.iterdir()]

        existing_skybox = self.mujoco_arena.asset.find("./texture[@type='skybox']")
        hdri_path = os.path.abspath(choice(hdris))
        
        rotations = [
            "RLUDFB",  # 0 deg
            "BFUDRL",  # 90 deg
            "LRUDBF",  # 180 deg
            "FBUDLR"   # 270 deg
        ]
        
        attribs = {
            "type": "skybox",
            "file": hdri_path,
            "gridsize": "1 6",
            "gridlayout": choice(rotations)
        }
        if existing_skybox is not None:
            existing_skybox.attrib.clear()
            existing_skybox.attrib.update(attribs)
        else:
            skybox = new_element("texture", **attribs)
            self.mujoco_arena.asset.append(skybox)
        
        table_visual = self.mujoco_arena.worldbody.find("./body[@name='table']/geom[@name='table_visual']")
        if table_visual is not None:
            table_visual.set("material", "desk_mat")

        frontview = self.mujoco_arena.worldbody.find("./camera[@name='frontview']")
        if frontview is not None:
            frontview.set("pos", "1.5 0 1.55") 
            frontview.set("quat", "0.56 0.33 0.33 0.56")

        import xml.etree.ElementTree as ET
        ET.SubElement(self.mujoco_arena.worldbody, "light", attrib={"pos": "0 0 1.5", "dir": "0 0 -1", "diffuse": "0.5 0.5 0.5"})

        # creating objects
        self.mujoco_objects = []
        self.objects_dict = {}

        for i, name in enumerate(self.scene_items):
            unique_name = name + '_' + str(i)
            config = self.item_registry[name]
            try:
                config['kwargs']['rgba'] = (random(), random(), random(), 1.0)
                obj = config['class'](name=unique_name, **config['kwargs'])
            except:
                del config['kwargs']['rgba']
                obj = config['class'](name=unique_name, **config['kwargs'])
                
            # choosing a random texture for the coaster(s), if spawned.
            if "coaster_a" in name:
                for tex in obj.asset.findall("./texture"):
                    if "T_Coaster_Color" in tex.get("name", ""):
                        pattern = choice(['pattern1.png', 'pattern2.png', 'pattern3.png', 'pattern4.png'])
                        tex.set("file", os.path.abspath(f"assets/models/coaster_a/{pattern}"))
                        
            self.mujoco_objects.append(obj)
            self.objects_dict[unique_name] = obj

        self.model = ManipulationTask(
            mujoco_arena=self.mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.mujoco_objects
        )

    def _setup_references(self):
        """Sets up references to important components/objects."""
        
        super()._setup_references()

        for name, obj in self.objects_dict.items():
            self.object_ids[name] = self.sim.model.body_name2id(obj.naming_prefix + "main")
        
    def _randomize_domain(self):
        """
        Applies domain randomization by varying lighting conditions, table colour, background and shadows.
        """
        model = self.sim.model

        table_geom_id = model.geom_name2id("table_visual")
        model.geom_rgba[table_geom_id] = [
            np.random.uniform(0.25, 0.95),
            np.random.uniform(0.25, 0.90),
            np.random.uniform(0.20, 0.85),
            1.0
        ]

        for i in range(model.nlight):
            intensity = np.random.uniform(0.3, 1.2)

            hue_tint = np.random.uniform(0.95, 1.05, 3)
            model.light_diffuse[i] = np.clip(intensity * hue_tint, 0.0, 1.0)
            model.light_specular[i] = np.clip(intensity * np.random.uniform(0.2, 0.5) * hue_tint, 0.0, 1.0)
            model.light_ambient[i] = np.random.uniform(0.0, 0.1, 3)

            model.light_dir[i] = np.array([
                np.random.uniform(-1.0, 1.0),
                np.random.uniform(-1.0, 1.0),
                np.random.uniform(-1.5, -0.3),
            ])

            model.light_pos[i] += np.random.uniform(-0.5, 0.5, 3)

            # occasionally toggle shadows to simulate hard lighting
            if np.random.random() < 0.5:
                model.light_castshadow[i] = 1
            else:
                model.light_castshadow[i] = 0

    def _reset_internal(self):
        """Sets initial placements."""
        
        self.sim.model.vis.quality.shadowsize = 8192
        
        super()._reset_internal()

        self._randomize_domain()

        coordinates = self._return_randoms(list(self.objects_dict.keys()))

        for i, (name, obj) in enumerate(self.objects_dict.items()):
            if obj.joints:
                # only z-orientation is randomized
                theta = np.random.uniform(0, 2 * np.pi)
                qw = np.cos(theta / 2)
                qz = np.sin(theta / 2)
                
                self.sim.data.set_joint_qpos(
                    obj.naming_prefix + "joint0", 
                    [coordinates[name][0], coordinates[name][1], self.height+0.05 , qw, 0, 0, qz]
                )
                time.sleep(0.03)
            else:
                self.sim.model.body_pos[self.object_ids[name]] = [coordinates[name][0], coordinates[name][1], self.height+0.001]

        self.sim.forward()

    def _check_success(self): # (not used in the current paradigm)
        return False

    def reward(self, action=None):
        return 0.0

    def render(self):
        super().render()

    def get_camera_image(self, camera_name: str, width: int = 256, height: int = 256) -> np.ndarray:
        img = self.sim.render(camera_name=camera_name, width=width, height=height, depth=False)
        img = np.flipud(img)[..., ::-1]
        return img

class DeskOrganizerEnv(gym.Env):
    """Gymnasium wrapper around DeskOrganizerRobosuiteEnv.
      - Enables gym registry and data collection with LeRobot's gym_manipulator.
      - Re-keys observations into {"pixels": {...}, "robot_state": {...}}
      - LeRobot's LiberoProcessorStep, etc handles conversions, preprocessing and postprocessing.
    """

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(self, task_desc="organize the items on the desk", render_mode=None,
                 image_obs=True, use_gripper=True, gripper_penalty=0.0,  make_vertical=False,
                 control_mode="auto", target_object="can", destination_object="holder_a", **kwargs):
        super().__init__()
        self.render_mode = render_mode
        self.task_description = task_desc
        self.use_gripper = use_gripper
        self.gripper_penalty = gripper_penalty
        self.control_mode = control_mode
        self.make_vertical = make_vertical
        
        # initialising path planner state for automated data collection
        self._bot_state = 0
        self._bot_wait_ticks = 0
        self._bot_done = False
        self._bot_grasp_z_offset = 0.0
        self._bot_lift_z_target = 0.0
        self._bot_open_time_offset = 0.0

        self._camera_name_mapping = {
            "frontview_image": "image",
            "robot0_eye_in_hand_image": "image2",
        }

        self._env = DeskOrganizerRobosuiteEnv(task_desc=task_desc,
                                              use_camera_obs=image_obs, 
                                              target_object=target_object, 
                                              destination_object=destination_object, 
                                              **kwargs)

        self._env.target_object = target_object
        self._env.destination_object = destination_object
        self._viewer = None
        
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "controller_config_new.json")
        self._gamepad = GamepadController(
            x_step_size=1.0, y_step_size=1.0, z_step_size=1.0,
            config_path=config_path,
        )
        self._gamepad.start()

        try:
            self._gamepad.update()
            self._has_gamepad = True
        except (AttributeError, Exception):
            self._has_gamepad = False
            print("No working gamepad detected, running in policy-only mode.")

        obs_h, obs_w = 256, 256
        images_space = gym.spaces.Dict({
            "image": gym.spaces.Box(0, 255, shape=(obs_h, obs_w, 3), dtype=np.uint8),
            "image2": gym.spaces.Box(0, 255, shape=(obs_h, obs_w, 3), dtype=np.uint8),
        })

        self.observation_space = gym.spaces.Dict({
            "pixels": images_space,
            "agent_pos": gym.spaces.Box(-np.inf, np.inf, shape=(9,), dtype=np.float32),
        })
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )

    @staticmethod
    def _augment_image(img):
        """
        Apply subtle image augmentation for sim-to-real robustness. Currently disabled.
        """
        img = img.astype(np.float32)

        # Gaussian noise (very subtle)
        sigma = np.random.uniform(1.0, 4.0)
        noise = np.random.normal(0, sigma, img.shape)
        img = img + noise

        # Color jitter (brightness, contrast, saturation) has been disabled.
        return np.clip(img, 0, 255).astype(np.uint8)

    def _format_raw_obs(self, raw_obs):
        """
        Re-key Robosuite obs dict into the format preprocess_observation expects.
        """
        images = {}
        for robosuite_key, lerobot_key in self._camera_name_mapping.items():
            img = np.flipud(raw_obs[robosuite_key]).copy()
            img = self._augment_image(img)
            images[lerobot_key] = img

        # Flatten EEF pose + gripper into a single 1D state vector (9 dims)
        # This is what pi0.5 read as "observation.state"
        agent_pos = np.concatenate([
            raw_obs["robot0_eef_pos"],       # (3,) xyz position
            raw_obs["robot0_eef_quat"],      # (4,) quaternion orientation
            raw_obs["robot0_gripper_qpos"],   # (2,) gripper finger positions
        ]).astype(np.float32)

        return {
            "pixels": images,
            "agent_pos": agent_pos,
        }

    @property
    def task(self):
        return self.task_description

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._env.seed(seed)
            
        self._bot_state = 0
        self._bot_wait_ticks = 0
        self._bot_done = False
        self._bot_start_eef_pos = None
        self._bot_grasp_z_offset = -0.005 #np.random.uniform(0.005, 0.015)
        self._bot_lift_z_target = np.random.uniform(0.2, 0.25)
        self._bot_open_time_offset = np.random.uniform(0.2, 0.7)
        self._bot_xy_jitter = np.zeros(2)
        self._bot_xy_jitter_target = np.random.uniform(-0.17, 0.17, size=2)
        self._bot_drop_offset = np.random.uniform(-0.005, 0.005, size=2)
        raw_obs = self._env.reset()

        for _ in range(5):
            raw_obs, _, _, _ = self._env.step(np.zeros(7))

        # create the mujoco glfw viewer AFTER reset
        if self.render_mode == "human":
            if self._viewer is not None:
                self._viewer.close()
            self._viewer = mujoco.viewer.launch_passive(
                self._env.sim.model._model, self._env.sim.data._data
            )
            self._viewer.opt.geomgroup[0] = 0
            self._viewer.opt.geomgroup[1] = 1
            for i in range(6):
                self._viewer.opt.sitegroup[i] = 0

        return self._format_raw_obs(raw_obs), {"is_success": False}
 
    def _path_planner(self):
        """State machine for automated demonstration collection."""
        target_bodies = [name for name in self._env.objects_dict.keys() if self._env.target_object in name]
        target_body_id = self._env.object_ids[target_bodies[0]]
        target_pos = self._env.sim.data.body_xpos[target_body_id].copy()
        
        if any(name in self._env.target_object for name in ["mug_a", "mug_b", "holder_a"]):
            target_pos[2] += 0.04
        
        destination_bodies = [name for name in self._env.objects_dict.keys() if self._env.destination_object in name]
        destination_body_id = self._env.object_ids[destination_bodies[0]]
        destination_pos = self._env.sim.data.body_xpos[destination_body_id].copy()
        
        eef_pos = self._env.sim.data.site_xpos[self._env.robots[0].eef_site_id]
        
        action = np.zeros(7, dtype=np.float32)
        action[6] = 1.0
        
        if self._env.timestep % 10 == 0:
            self._bot_xy_jitter_target = np.random.uniform(-0.02, 0.02, size=2)
        self._bot_xy_jitter += (self._bot_xy_jitter_target - self._bot_xy_jitter) * 0.15
        
        def compute_dpos(target, current, speed=1.0, apply_jitter=False):
            delta = target - current
            if apply_jitter:
                delta[:2] += self._bot_xy_jitter
                
            norm = np.linalg.norm(delta)
            if norm < 0.01:
                return np.zeros(3)
            return (delta / norm) * min(norm * 5.0, speed)
            
        speed = 0.8
        
        if self._bot_state == 0:
            if self._bot_start_eef_pos is None:
                self._bot_start_eef_pos = eef_pos.copy()
                
            approach_dir = target_pos[:2] - self._bot_start_eef_pos[:2]
            norm_dir = np.linalg.norm(approach_dir)
            if norm_dir > 0.001:
                approach_dir = approach_dir / norm_dir
            else:
                approach_dir = np.zeros(2)
                
            overshoot_mm = 0.005  # move 5mm extra in the initial approach direction
            
            # STATE 0: hover directly above the target object
            hover_target = np.array([target_pos[0] + approach_dir[0] * overshoot_mm, 
                                     target_pos[1] + approach_dir[1] * overshoot_mm, 
                                     eef_pos[2]])
            action[:3] = compute_dpos(hover_target, eef_pos, speed=speed, apply_jitter=True)
            
            dist_to_target = np.linalg.norm(hover_target[:2] - eef_pos[:2])
            if dist_to_target < self._bot_open_time_offset:
                action[6] = 0.0
                
            if dist_to_target < 0.02:
                self._bot_state = 1
                
        elif self._bot_state == 1:
            if self._env.target_object in ["pen_a", "highlighter", "pencil", "highlighter", "spoon", "mouse", "calculator", "mug_b", "mug_a"]:
            # STATE 1: rotate the gripper to align parallel with the object's long axis (skipped if the object is symmetric)
                action[6] = 0.0

                target_rmat = np.array(self._env.sim.data.body_xmat[target_body_id].reshape(3,3), dtype=np.float32)
                eef_rmat = np.array(self._env.sim.data.site_xmat[self._env.robots[0].eef_site_id].reshape(3,3), dtype=np.float32)

                v_target = target_rmat[:2, 0]
                v_eef = eef_rmat[:2, 0]
                v_target = v_target / (np.linalg.norm(v_target) + 1e-6)
                v_eef = v_eef / (np.linalg.norm(v_eef) + 1e-6)

                cross_prod = v_eef[0]*v_target[1] - v_eef[1]*v_target[0]
                dot_prod = v_eef[0]*v_target[0] + v_eef[1]*v_target[1]
                dyaw = np.arctan2(cross_prod, dot_prod)

                #dyaw = dyaw - np.pi/2
                dyaw = (dyaw + np.pi/2) % np.pi - np.pi/2

                local_z = eef_rmat[:, 2]
                action[3:6] = local_z * (-np.clip(dyaw * 1.5, -0.4, 0.4))

                if abs(dyaw) < 0.05:
                    self._bot_wait_ticks += 1
                    if self._bot_wait_ticks > 5:
                        self._bot_state = 2
                        self._bot_wait_ticks = 0
                else:
                    self._bot_wait_ticks = 0

            else:
                self._bot_state = 2
        elif self._bot_state == 2:
            # STATE 2: descend down to the object to grasp it
            action[6] = 0.0
            descend_target = np.array([target_pos[0], target_pos[1], target_pos[2] + self._bot_grasp_z_offset])
            action[:3] = compute_dpos(descend_target, eef_pos, speed=speed)
            
            if abs(eef_pos[2] - getattr(self, "_last_eef_z", 0.0)) < 0.0005:
                self._bot_wait_ticks += 1
            else:
                self._bot_wait_ticks = 0
            self._last_eef_z = eef_pos[2]
            
            # transition if reached target OR stuck against the table for 0.5s (10 ticks)
            if abs(descend_target[2] - eef_pos[2]) < 0.01 or self._bot_wait_ticks > 10:
                self._bot_state = 3
                self._bot_wait_ticks = 0
                
        elif self._bot_state == 3:
            # STATE 3: close the gripper fingers securely
            action[6] = 2.0
            self._bot_wait_ticks += 1
            if self._bot_wait_ticks > 15:
                self._bot_state = 4
                self._bot_wait_ticks = 0
                self._bot_grasp_z = eef_pos[2]
                
        elif self._bot_state == 4:
            # STATE 4: lift the grasped object up into the air
            action[6] = 2.0
            lift_target = np.array([eef_pos[0], eef_pos[1], destination_pos[2] + self._bot_lift_z_target])
            action[:3] = compute_dpos(lift_target, eef_pos, speed=speed)
            
            if abs(lift_target[2] - eef_pos[2]) < 0.01:
                self._bot_state = 5
                
        elif self._bot_state == 5:
            # STATE 5: carry the object over to the destination area
            action[6] = 2.0
            hover_circle = np.array([destination_pos[0] + self._bot_drop_offset[0], 
                                     destination_pos[1] + self._bot_drop_offset[1], 
                                     eef_pos[2]])
            
            dist = np.linalg.norm(hover_circle[:2] - eef_pos[:2])
            action[:3] = compute_dpos(hover_circle, eef_pos, speed=speed, apply_jitter=False)
            
            if dist < 0.01:
                self._bot_state = 6

        elif self._bot_state == 6:
            # STATE 6: descend slightly before making vertical or dropping
            if self._env.destination_object in ["coaster_a", "pad"]:
                action[6] = 2.0
                target_z = getattr(self, "_bot_grasp_z", eef_pos[2]) + 0.02
                #descend_target = np.array([eef_pos[0], eef_pos[1], target_z])
                descend_target = np.array([destination_pos[0] + self._bot_drop_offset[0],
                                           destination_pos[1] + self._bot_drop_offset[1], 
                                           target_z])
                action[:3] = compute_dpos(descend_target, eef_pos, speed=speed)
                
                if abs(eef_pos[2] - getattr(self, "_last_eef_z", 0.0)) < 0.0005:
                    self._bot_wait_ticks += 1
                else:
                    self._bot_wait_ticks = 0
                self._last_eef_z = eef_pos[2]
                
                if abs(descend_target[2] - eef_pos[2]) < 0.01 or self._bot_wait_ticks > 10:
                    self._bot_state = 7
                    self._bot_wait_ticks = 0
            else:
                action[6] = 2.0
                if not hasattr(self, "_bot_descend_target_z"):
                    self._bot_descend_target_z = eef_pos[2] - 0.03
                
                descend_target = np.array([eef_pos[0], eef_pos[1], self._bot_descend_target_z])
                action[:3] = compute_dpos(descend_target, eef_pos, speed=speed)
                
                if abs(eef_pos[2] - getattr(self, "_last_eef_z", 0.0)) < 0.0005:
                    self._bot_wait_ticks += 1
                else:
                    self._bot_wait_ticks = 0
                self._last_eef_z = eef_pos[2]
                
                if abs(descend_target[2] - eef_pos[2]) < 0.01 or self._bot_wait_ticks > 10:
                    self._bot_state = 7
                    self._bot_wait_ticks = 0
                    del self._bot_descend_target_z
                    
        elif self._bot_state == 7:
            # STATE 7: make vertical - rotate until object's x axis is perpendicular to the table. only for tasks involving putting narrow objects in containers.
            action[6] = 2.0
            
            if not hasattr(self, "_bot_rotation_z"):
                self._bot_rotation_z = eef_pos[2]
            
            # # Keep position stable while rotating
            # dest_body_name = [name for name in self._env.sim.model.body_names if self._env.destination_object in name][0]
            # hover_circle = np.array([self._env.sim.data.body_xpos[self._env.sim.model.body_name2id(dest_body_name)][0] + self._bot_drop_offset[0], 
            #                          self._env.sim.data.body_xpos[self._env.sim.model.body_name2id(dest_body_name)][1] + self._bot_drop_offset[1], 
            #                          self._bot_rotation_z])
            # action[:3] = compute_dpos(hover_circle, eef_pos, speed=speed)

            hover_circle = np.array([destination_pos[0] + self._bot_drop_offset[0], 
                                     destination_pos[1] + self._bot_drop_offset[1], 
                                     eef_pos[2]])
            
            dist = np.linalg.norm(hover_circle[:2] - eef_pos[:2])
            action[:3] = compute_dpos(hover_circle, eef_pos, speed=speed, apply_jitter=False)

            if self.make_vertical:
                target_bodies = [name for name in self._env.sim.model.body_names if self._env.target_object in name]
                target_body_id = self._env.sim.model.body_name2id(target_bodies[0])
                target_rmat = np.array(self._env.sim.data.body_xmat[target_body_id].reshape(3,3), dtype=np.float32)
                
                target_long_axis = target_rmat[:, 1]
                z_component = target_long_axis[2]
                
                if abs(z_component) > 0.72 or getattr(self, "_bot_state7_ticks", 0) > 80:
                    self._bot_wait_ticks += 1
                    if self._bot_wait_ticks > 5:
                        action[3:6] = np.zeros(3)
                        xy_err = np.linalg.norm(hover_circle[:2] - eef_pos[:2])
                        if xy_err < 0.015:
                            self._bot_state = 8
                        else:
                            self._bot_state = 5
                        self._bot_wait_ticks = 0
                        self._bot_state7_ticks = 0
                        del self._bot_rotation_z
                else:
                    self._bot_state7_ticks = getattr(self, "_bot_state7_ticks", 0) + 1
                    self._bot_wait_ticks = 0
                    xy_err = np.linalg.norm(hover_circle[:2] - eef_pos[:2])
                    if xy_err < 0.015:
                        desired_dir = np.array([0.0, 0.0, 1.0]) if z_component > 0 else np.array([0.0, 0.0, -1.0])
                        rot_axis = np.cross(target_long_axis, desired_dir)
                        rot_axis_norm = np.linalg.norm(rot_axis)
                        
                        if rot_axis_norm > 5e-3:
                            action[3:6] = (rot_axis / rot_axis_norm) * 1
            else:
                self._bot_state = 8
                self._bot_wait_ticks = 0
                del self._bot_rotation_z
                
        elif self._bot_state == 8:
            # STATE 8: open the gripper to drop the object and finish
            action[6] = 0.0
            self._bot_wait_ticks += 1
            if self._bot_wait_ticks > 40:
                self._bot_state = 9
                self._bot_done = True
                
        return action

    def step(self, action):
        """Steps the environment and introduces gamepad/automated teleoperation commands, if enabled."""
        action = np.asarray(action, dtype=np.float32)

        is_intervention = False
        episode_end_status = None
        gamepad_action = np.zeros(4, dtype=np.float32)

        if self._has_gamepad:
            try:
                self._gamepad.update()
                is_intervention = not self._gamepad.should_intervene()
                episode_end_status = self._gamepad.get_episode_end_status()
            except (AttributeError, Exception):
                pass  # gamepad disconnected mid-run, ignore

            if is_intervention:
                if self.control_mode == "manual":
                    delta_x, delta_y, delta_z = self._gamepad.get_deltas()
                    gamepad_action = np.array([delta_x, delta_y, delta_z], dtype=np.float32)
                    if self.use_gripper:
                        gripper_cmd = self._gamepad.gripper_command()
                        if gripper_cmd == "open":
                            gamepad_action = np.concatenate([gamepad_action, [2.0]])
                        elif gripper_cmd == "close":
                            gamepad_action = np.concatenate([gamepad_action, [0.0]])
                        else:
                            gamepad_action = np.concatenate([gamepad_action, [1.0]])
                elif self.control_mode == "auto":
                    gamepad_action = self._path_planner()
                    
                action = gamepad_action
            
        if self._bot_done and self.control_mode == "auto":
            episode_end_status = "success"

        terminate_episode = episode_end_status is not None
        success = episode_end_status == "success"
        rerecord_episode = episode_end_status == "rerecord_episode"

        # gym_manipulator sends 4D actions: [dx, dy, dz, gripper]
        # 7D: [dx, dy, dz, drot_x, drot_y, drot_z, gripper]
        if action.shape[-1] == 7:
            pos = action[:3] * EE_STEP_SIZE
            rot = action[3:6]
            grip = np.array([action[6] - 1.0])
            action = np.concatenate([pos, rot, grip])
            #pass
            # the above lines must be disabled during inference.
        elif action.shape[-1] == 4:
            pos = action[:3] * EE_STEP_SIZE
            rot = np.zeros(3, dtype=np.float32)    # no rotation control
            grip = np.array([action[3] - 1.0])     # map [0,2] → [-1,1]
            action = np.concatenate([pos, rot, grip])

        executed_action = action[:7].copy()
        raw_obs, reward, done, info = self._env.step(action[:7])

        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()

        is_success = self._env._check_success()
        terminated = done or is_success or terminate_episode
        truncated = False

        if success:
            reward = 1.0

        info.update({
            "is_success": is_success,
            "is_intervention": is_intervention,
            "teleop_action": gamepad_action,
            "executed_action": executed_action,
            "rerecord_episode": rerecord_episode,
        })
        return self._format_raw_obs(raw_obs), reward, terminated, truncated, info

    def render(self):
        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()
        return self._env.render()

    def close(self):
        if hasattr(self, '_gamepad'):
            self._gamepad.stop()
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        self._env.close()
    

if __name__ == "__main__":
    import time

    os.environ['MUJOCO_GL'] = 'glx'
    
    print("Initializing DeskOrganizerEnv...")
    
    env = DeskOrganizerEnv(
        has_renderer=True,
        render_camera='frontview',
        has_offscreen_renderer=True,
        image_obs=True,
        control_freq=20,
        horizon=500,
        target_object='calculator',
        destination_object='pad',
        make_vertical=False,
    )
    obs = env.reset()
    
    viewer = mujoco.viewer.launch_passive(env._env.sim.model._model, env._env.sim.data._data)
    viewer.opt.geomgroup[0] = 0
    viewer.opt.geomgroup[1] = 1
    
    viewer.cam.azimuth = 140
    viewer.cam.elevation = -40
    viewer.cam.distance = 1.0
    viewer.cam.lookat[:] = [0.1, 0.0, 0.85]

    print("Running simulation loop. Press Ctrl+C to terminate.")
    start = time.time()
    try:
        while True:
            current = time.time()
            if (current-start) % 4 < 1:
                dx,dy = 0,1
            elif (current-start) % 4 < 2:
                dx,dy = 1,0
            elif (current-start) % 4 < 3:
                dx,dy = 0,-1
            else:
                dx,dy = -1,0

            action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
            
            obs, reward, terminated, truncated, info = env.step(action) # add truncated if using gym wrapper
            done = terminated or truncated
            
            if viewer is not None and viewer.is_running():
                viewer.sync()
            
            #env.render()
            
            time.sleep(1.0 / env._env.control_freq)
            
            if done:
                print("Episode finished. Resetting...")
                env.reset()
                
    except KeyboardInterrupt:
        print("\nTerminating simulation...")
    finally:
        if viewer is not None:
            viewer.close()
        env.close()
        print("Environment closed cleanly.")

