import gymnasium
from gymnasium import spaces
import numpy as np
import random
import genesis as gs
import pathlib as pl
import cv2
import torch
from kinova import JOINT_NAMES as kinova_joint_names, EEF_NAME as kinova_eef_name, TRIALS_POSITION_0, TRIALS_POSITION_1, TRIALS_POSITION_2

FINGERTIP_POS = 1.0
KINOVA_START_DOFS_POS = [0.3268500269015339, -1.4471734542578538, 2.3453266624159497, -1.3502152158191212, 2.209384006676201, -1.5125125137062945, -1, 1, FINGERTIP_POS, FINGERTIP_POS]
PX, PZ = 0.465, 0.05
POSITION_0 = torch.tensor((PX, 0.1, PZ))
POSITION_1 = torch.tensor((PX, -0.05, PZ))
POSITION_2 = torch.tensor((PX, -0.2, PZ))

def _normalize_action(action):
    """
    Normalize the action from the action space to the range [-1, 1].
    """
    action_space = GenesisGym.action_space
    action = (action - action_space.low) / (action_space.high - action_space.low)
    return 2 * action - 1

def _unnormalize_action(action, action_space):
    """
    Unnormalize the action from the range [-1, 1] to the action space.
    """
    action = (action + 1) / 2 * (action_space.high - action_space.low) + action_space.low
    return action


class GenesisGym(gymnasium.Env):
    """
    Custom Gymnasium environment for the Genesis game.
    """
    
    # make a class wide action space
    # Actions are 7 continuous actions. 6 dof joint angles, 1 gripper position
    action_space = spaces.Box(low=np.array([-3.14, -3.14, -3.14, -3.14, -3.14, -3.14, 0]), high=np.array([3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 100.]), shape=(7,), dtype=np.float32)
    
    def __init__(self, args={}, size=(96, 96), use_truncated_in_return=False):
        super().__init__()
        self.args = {
            'rho': args.rho if hasattr(args, 'rho') else DEFAULT_RHO,
            'radius': args.radius if hasattr(args, 'radius') else DEFAULT_RADIUS,
            'height': args.height if hasattr(args, 'height') else DEFAULT_HEIGHT,
            'friction': args.friction if hasattr(args, 'friction') else DEFAULT_FRICTION,
            'vis': args.vis if hasattr(args, 'vis') else False,
            # 'starting_x': args.starting_x if hasattr(args, 'starting_x') else 0.65
            }

        self.size = size
        self.n_steps = 0
        # Define action and observation space
        # Observations are either an image, a state, or a combination
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(*size, 3), dtype=np.uint8),
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(10 + 3,), dtype=np.float32), # joint angles and gripper state as well as can location and differential to goal
            'reward': spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32),
            'is_first': spaces.Box(low=0, high=1, shape=(), dtype=bool),
            'is_last': spaces.Box(low=0, high=1, shape=(), dtype=bool),
            'is_terminal': spaces.Box(low=0, high=1, shape=(), dtype=bool),
        })

        self.last_arm_dofs = None

        gs.init(backend=gs.cpu, seed=random.randint(0, 2**30), precision="32", logging_level="warning")
        self.metadata = {
            "render_fps": 30
        }

        self.use_truncated_in_return = use_truncated_in_return
        self.force_sparse = True

        self.init_env()


    def _max_episode_steps(self):
        return 100

    def init_env(self):
        pass

    def step(self, action):
        # Apply the action to the scene
        self.apply_action(action)
        self.scene.step()
        obs = self.get_obs()
        reward = obs['reward']
        done = obs['is_last']

        self.n_steps += 1

        if self.use_truncated_in_return:
            return obs, reward, done, self.n_steps >= self._max_episode_steps(), {'is_success': done}
        return obs, reward, done, {}
    
    def reset(self, trial_id=0, **kwargs):
        pass
    
    def get_obs(self, is_first=False):
        # Get the current observation from the scene
        image = self.cam_0.render(rgb=True, depth=False, segmentation=False, normal=False)
        image = image[0] # grab the rgb
        # resize the image to the desired size
        image = cv2.resize(image, self.size)

        arm_pos = self.kinova.get_dofs_position(dofs_idx_local=self.kdofs_idx).cpu().numpy()
        teapot_pos = self.mug.get_pos().cpu().numpy()
        state = np.concatenate((arm_pos, teapot_pos))

        self.last_arm_dofs = arm_pos

        reward, done = self.compute_reward(state)
        return {"image": image, "state": state, "reward": reward, "is_first": is_first, "is_last": done, "is_terminal": False}
    
    def calc_gripper_force(self, cmd_gripper_pos, threshold=0.03):
        # Calculate the gripper force based on the gripper position
        pos = self.last_arm_dofs
        output_force = [0., 0., 0., 0.]
        motor_cmd = (100 - cmd_gripper_pos) / 100
        right_error = pos[-4] + motor_cmd; right_error = right_error if abs(right_error) > threshold else [0.0]
        left_error = pos[-3] - motor_cmd; left_error = left_error if abs(left_error) > threshold else [0.0]
        right_fingertip_error = pos[-2] - KINOVA_START_DOFS_POS[-2]; right_fingertip_error = right_fingertip_error if abs(right_fingertip_error) > threshold else 0.0
        left_fingertip_error = pos[-1] - KINOVA_START_DOFS_POS[-1]; left_fingertip_error = left_fingertip_error if abs(left_fingertip_error) > threshold else 0.0

        output_force[0] = -self.kp*right_error[0]; output_force[2] = self.kp*right_fingertip_error
        output_force[1] = -self.kp*left_error[0]; output_force[3] = self.kp*left_fingertip_error
        # print(output_force)
        return np.array(output_force)

    def apply_action(self, action):

        arm_pos, gripper_pos = action[:6], action[6:]

        gripper_force = self.calc_gripper_force(gripper_pos)

        self.kinova.control_dofs_force(gripper_force, dofs_idx_local=np.array(self.kdofs_idx[-4:]))
        self.kinova.control_dofs_position(arm_pos, dofs_idx_local=self.kdofs_idx[:len(arm_pos)])

    def compute_reward(self, obs):
        pass

    def render(self, mode='human', use_imshow=False):
        # Render the scene
        img = None
        if mode == 'human':
            img = self.cam_0.render(rgb=True, depth=False, segmentation=False, normal=False, use_imshow=False)[0]
            img = cv2.resize(img, self.size)
            if use_imshow:
                cv2.imshow('Genesis Gym', img)
                cv2.waitKey(1)
        return img
    