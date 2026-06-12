import gymnasium as gym
import numpy as np
import os
import cv2
import mujoco
from gymnasium.envs.registration import register
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.tasks import ManipulationTask
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject, CylinderObject, MujocoXMLObject
from robosuite.models.objects import MujocoXMLObject
from robosuite.controllers import load_controller_config
from robosuite.utils.mjcf_utils import new_element
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
        self.task_desc = kwargs.pop("task_desc", "organize the items on the desk")
        # Configure the Panda arm, gripper, and workspace
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
        from robosuite.controllers import load_controller_config
        ctrl_config = load_controller_config(default_controller="OSC_POSE")
        ctrl_config["kp"] = 2000  # increased from default 150 
        kwargs.setdefault("controller_configs", ctrl_config)
        self.height=0.9
        super().__init__(**kwargs)

    def _return_randoms(self, items: list, x_max=0.3, x_min=0.0, y_max=0.4, y_min=-0.4, margin=0.1):
        """
        Returns a set of random coordinates to place objects, separated by a minimum distance.
        """
        coords = {}
        for item in items:
            too_close = True
            while too_close:
                too_close = False
                new = np.array([np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max)])
                for pair in coords.values():
                    if np.linalg.norm(pair - new) < margin:
                        too_close = True
            coords[item] = new
        return coords
        

    def _load_model(self):
        super()._load_model()
        
        self.table_full_size = (0.5, 1, 0.05)
        self.table_offset = np.array((0.15, 0, self.height))

        # Adjust the robot's base position so it sits on the table instead of the ground floor
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)
        
        self.mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=(1.0, 0.005, 0.0001),
            table_offset=self.table_offset
        )
        self.mujoco_arena.set_origin([0, 0, 0])

        # Applying checkered texture to the table
        
        grid_tex = new_element("texture", name="desk_grid", type="2d", builtin="checker", 
                               rgb1="0.6 0.6 0.6", rgb2="0.5 0.5 0.5", width="512", height="512")
        grid_mat = new_element("material", name="desk_mat", texture="desk_grid", 
                               texrepeat="10 10", texuniform="true", reflectance="0.0")
        
        self.mujoco_arena.asset.append(grid_tex)
        self.mujoco_arena.asset.append(grid_mat)
        
        table_visual = self.mujoco_arena.worldbody.find("./body[@name='table']/geom[@name='table_visual']")
        if table_visual is not None:
            table_visual.set("material", "desk_mat")

        # Adjusting front view camera

        frontview = self.mujoco_arena.worldbody.find("./camera[@name='frontview']")
        if frontview is not None:
            frontview.set("pos", "1.5 0 1.55") 
            frontview.set("quat", "0.56 0.33 0.33 0.56")

        # Creating objects

        self.circle = CylinderObject(
            name="circle",
            size_min=(0.05, 0.001), # radius, half-height
            size_max=(0.05, 0.001),
            rgba=(0.8, 0.2, 0.2, 1.0),
            friction=(1.0, 0.005, 0.0001),
            joints=None
        )
        
        self.blue_cube = BoxObject(
            name="blue_cube",
            size=(0.025, 0.025, 0.025), # half-sizes
            rgba=(0.2, 0.2, 0.8, 1.0)
        )

        self.yellow_cube = BoxObject(
            name="yellow_cube",
            size=(0.025, 0.025, 0.025), # half-sizes
            rgba=(0.8, 0.8, 0.2, 1.0)
        )

        self.model = ManipulationTask(
            mujoco_arena=self.mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.circle, self.blue_cube, self.yellow_cube]
        )

    def _setup_references(self):
        """Sets up references to important components/objects."""
        
        super()._setup_references()
        
        self.circle_body_id = self.sim.model.body_name2id(self.circle.naming_prefix + "main")
        self.blue_cube_body_id = self.sim.model.body_name2id(self.blue_cube.naming_prefix + "main")
        self.yellow_cube_body_id = self.sim.model.body_name2id(self.yellow_cube.naming_prefix + "main")
    
    def _reset_internal(self):
        """Sets initial placements."""
        
        super()._reset_internal()

        coordinates = self._return_randoms(['blue_cube', 'yellow_cube', 'circle'])
        
        circle_x = coordinates['circle'][0]
        circle_y = coordinates['circle'][1]

        self.sim.data.set_joint_qpos(
            self.blue_cube.naming_prefix + "joint0", 
            [coordinates['blue_cube'][0], coordinates['blue_cube'][1], self.height+0.02, 0, 0, 0, 1]
        )

        self.sim.data.set_joint_qpos(
            self.yellow_cube.naming_prefix + "joint0", 
            [coordinates['yellow_cube'][0], coordinates['yellow_cube'][1], self.height+0.02, 0, 0, 0, 1]
        )


        self.sim.model.body_pos[self.circle_body_id] = [circle_x, circle_y, self.height+0.001]

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
    """Thin Gymnasium wrapper around DeskOrganizerRobosuiteEnv.
      - Enables gym registry and data collection with LeRobot's gym_manipulator.
      - Re-keys observations into {"pixels": {...}, "robot_state": {...}}
      - LeRobot's LiberoProcessorStep, etc handles conversions, preprocessing and postprocessing.
    """

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(self, task_desc="organize the items on the desk", render_mode=None,
                 image_obs=True, use_gripper=True, gripper_penalty=0.0, 
                 control_mode="auto", target_object="blue_cube", **kwargs):
        super().__init__()
        self.render_mode = render_mode
        self.task_description = task_desc
        self.use_gripper = use_gripper
        self.gripper_penalty = gripper_penalty
        self.control_mode = control_mode
        self.target_object = target_object
        
        # Path Planner State (for automated data collection in the simulation)
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

        self._env = DeskOrganizerRobosuiteEnv(task_desc=task_desc, use_camera_obs=image_obs, **kwargs)

        self._viewer = None
        
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "controller_config_new.json")
        self._gamepad = GamepadController(
            x_step_size=1.0, y_step_size=1.0, z_step_size=1.0,
            config_path=config_path,
        )
        self._gamepad.start()

        # Test if the gamepad actually works
        try:
            self._gamepad.update()
            self._has_gamepad = True
        except (AttributeError, Exception):
            self._has_gamepad = False
            print("[INFO] No working gamepad detected — running in policy-only mode.")

        # Define Gymnasium spaces
        obs_h, obs_w = 256, 256
        images_space = gym.spaces.Dict({
            "image": gym.spaces.Box(0, 255, shape=(obs_h, obs_w, 3), dtype=np.uint8),
            "image2": gym.spaces.Box(0, 255, shape=(obs_h, obs_w, 3), dtype=np.uint8),
        })
        # Flat state: [eef_pos(3), eef_quat(4), gripper_qpos(2)] = 9 dimensions
        self.observation_space = gym.spaces.Dict({
            "pixels": images_space,
            "agent_pos": gym.spaces.Box(-np.inf, np.inf, shape=(9,), dtype=np.float32),
        })
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )

    def _format_raw_obs(self, raw_obs):
        """
        Re-key Robosuite obs dict into the format preprocess_observation expects.
        """
        images = {}
        for robosuite_key, lerobot_key in self._camera_name_mapping.items():
            # OpenGL returns images upside down, so we flip them vertically
            images[lerobot_key] = np.flipud(raw_obs[robosuite_key]).copy()

        # Flatten EEF pose + gripper into a single 1D state vector (9 dims)
        # This is what Pi0/Pi0.5 read as "observation.state"
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
        self._bot_grasp_z_offset = np.random.uniform(-0.01, 0.01)
        self._bot_lift_z_target = np.random.uniform(0.1, 0.15)
        self._bot_open_time_offset = np.random.uniform(0.2, 0.7)
        self._bot_xy_jitter = np.zeros(2)
        self._bot_xy_jitter_target = np.random.uniform(-0.17, 0.17, size=2)
        self._bot_drop_offset = np.random.uniform(-0.015, 0.015, size=2)
        
        raw_obs = self._env.reset()

        for _ in range(5):
            raw_obs, _, _, _ = self._env.step(np.zeros(7))

        # (Re-)create the MuJoCo GLFW viewer AFTER reset
        if self.render_mode == "human":
            if self._viewer is not None:
                self._viewer.close()
            self._viewer = mujoco.viewer.launch_passive(
                self._env.sim.model._model, self._env.sim.data._data
            )
            # Hide collision geoms, show only visual meshes
            self._viewer.opt.geomgroup[0] = 0
            self._viewer.opt.geomgroup[1] = 1
            # Hide site visualizations
            for i in range(6):
                self._viewer.opt.sitegroup[i] = 0

        return self._format_raw_obs(raw_obs), {"is_success": False}
 
    def _path_planner(self):
        """State machine for automated demonstration collection."""
        target_body_id = self._env.blue_cube_body_id if self.target_object == "blue_cube" else self._env.yellow_cube_body_id
        target_pos = self._env.sim.data.body_xpos[target_body_id]
        circle_pos = self._env.sim.data.body_xpos[self._env.circle_body_id]
        eef_pos = self._env.sim.data.site_xpos[self._env.robots[0].eef_site_id]
        
        action = np.zeros(4, dtype=np.float32)
        action[3] = 1.0
        
        # Smoothly wander the XY jitter to mimic human joystick drift
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
            hover_target = np.array([target_pos[0], target_pos[1], eef_pos[2]])
            action[:3] = compute_dpos(hover_target, eef_pos, speed=speed, apply_jitter=True)
            
            dist_to_target = np.linalg.norm(hover_target[:2] - eef_pos[:2])
            if dist_to_target < self._bot_open_time_offset:
                action[3] = 0.0
                
            if dist_to_target < 0.02:
                self._bot_state = 1
                
        elif self._bot_state == 1:
            action[3] = 0.0
            self._bot_wait_ticks += 1
            if self._bot_wait_ticks > 1:
                self._bot_state = 2
                self._bot_wait_ticks = 0
                
        elif self._bot_state == 2:
            action[3] = 0.0
            descend_target = np.array([target_pos[0], target_pos[1], target_pos[2] + self._bot_grasp_z_offset])
            action[:3] = compute_dpos(descend_target, eef_pos, speed=speed)
            
            if abs(descend_target[2] - eef_pos[2]) < 0.01:
                self._bot_state = 3
                
        elif self._bot_state == 3:
            action[3] = 2.0
            self._bot_wait_ticks += 1
            if self._bot_wait_ticks > 6:
                self._bot_state = 4
                self._bot_wait_ticks = 0
                
        elif self._bot_state == 4:
            action[3] = 2.0
            lift_target = np.array([eef_pos[0], eef_pos[1], circle_pos[2] + self._bot_lift_z_target])
            action[:3] = compute_dpos(lift_target, eef_pos, speed=speed)
            
            if abs(lift_target[2] - eef_pos[2]) < 0.01:
                self._bot_state = 5
                
        elif self._bot_state == 5:
            action[3] = 2.0
            hover_circle = np.array([circle_pos[0] + self._bot_drop_offset[0], 
                                     circle_pos[1] + self._bot_drop_offset[1], 
                                     eef_pos[2]])
            
            dist = np.linalg.norm(hover_circle[:2] - eef_pos[:2])
            # Turn off jitter when within 4cm
            action[:3] = compute_dpos(hover_circle, eef_pos, speed=speed, apply_jitter=(dist > 0.04))
            
            # Require it to get close before dropping
            if dist < 0.015:
                self._bot_state = 6
                
        elif self._bot_state == 6:
            action[3] = 0.0
            self._bot_wait_ticks += 1
            if self._bot_wait_ticks > 8:
                self._bot_state = 7
                self._bot_done = True
                
        return action

    def step(self, action):
        """Steps the environment and introduces gamepad/automated teleoperation commands, if enabled."""
        action = np.asarray(action, dtype=np.float32)
        self._env.timestep += 1

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
        # robosuite OSC_POSE expects 7D: [dx, dy, dz, drot_x, drot_y, drot_z, gripper]
        if action.shape[-1] == 4:
            pos = action[:3] * EE_STEP_SIZE
            rot = np.zeros(3, dtype=np.float32)    # no rotation control
            grip = np.array([action[3] - 1.0])     # map [0,2] → [-1,1]
            action = np.concatenate([pos, rot, grip])

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
    
    # has_renderer=True for opencv rendering
    env = DeskOrganizerRobosuiteEnv(
        has_renderer=True,
        render_camera='frontview',
        has_offscreen_renderer=True,
        use_camera_obs=True,
        control_freq=20,
        horizon=500
    )
    
    obs = env.reset()
    
    # mujoco 3d viewer
    print("Launching native MuJoCo GLFW passive viewer...")
    viewer = mujoco.viewer.launch_passive(env.sim.model._model, env.sim.data._data)
    viewer.opt.geomgroup[0] = 0
    viewer.opt.geomgroup[1] = 1  # Ensure visual meshes are enabled

    print("Running simulation loop. Press Ctrl+C to terminate.")
    start = time.time()
    try:
        while True:
            # circular motion
            current = time.time()
            if (current-start) % 4 < 1:
                dx,dy = 0,1
            elif (current-start) % 4 < 2:
                dx,dy = 1,0
            elif (current-start) % 4 < 3:
                dx,dy = 0,-1
            else:
                dx,dy = -1,0
            #dx = 0.5 * np.sin(t_seconds)
            #dy = 0.5 * np.cos(t_seconds)

            action = np.array([dx, dy, 0.0, 0.0, 0.0, 0.0, -1.0]) # Gripper closed
            #print(action, ' ||||||| ', action.dtype)
            
            # Step the simulation (Gymnasium returns 5-tuple)
            obs, reward, terminated, info = env.step(action) # add truncated if using gym wrapper
            done = terminated #or truncated
            
            # 3d viewer update
            if viewer is not None and viewer.is_running():
                viewer.sync()
            
            # opencv cam render
            env.render()
            
            # Maintain a realistic timestep rate
            time.sleep(1.0 / env.control_freq)
            
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

