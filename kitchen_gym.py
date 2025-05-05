# Adapted from https://medium.com/@thechrisyoon/deriving-policy-gradients-and-implementing-reinforce-f887949bd63
import matplotlib.pyplot as plt
from genesis_gym import GenesisGym
import numpy as np
import random
import genesis as gs
import pathlib as pl
import torch
from kinova import JOINT_NAMES as kinova_joint_names, EEF_NAME as kinova_eef_name, TRIALS_POSITION_0, TRIALS_POSITION_1, TRIALS_POSITION_2

FINGERTIP_POS = 1.0
KINOVA_START_DOFS_POS = [0.3268500269015339, -1.4471734542578538, 2.3453266624159497, -1.3502152158191212, 2.209384006676201, -1.5125125137062945, -1, 1, FINGERTIP_POS, FINGERTIP_POS]
MUG_POSITION = torch.tensor((0.65, -0.225, 0.17))
TEAPOT_POSITION = torch.tensor((0.7, 0, 0.17))
TABLE_WIDTH, TABLE_HEIGHT = 0.75, 0.14

# PX, PZ = 0.7, 0.17
# POSITION_0 = torch.tensor((PX, -0.1, PZ))
# POSITION_1 = torch.tensor((PX, -0.05, PZ))
# POSITION_2 = torch.tensor((PX, -0.2, PZ))
PX, PZ = 0.7, 0.17
POSITION_0 = torch.tensor((PX, -0.1, PZ))
POSITION_1 = torch.tensor((PX, -0.1, PZ))
POSITION_2 = torch.tensor((PX, -0.1, PZ))



class KitchenGym(GenesisGym):
    """
    Custom Gymnasium environment for the Genesis game.
    """
    def __init__(self, args={}, size=(96, 96), use_truncated_in_return=False):
        super().__init__(args)
        self.successes = []

    def init_env(self):
        self.kp = kp = 5

        # self.scene = scene = gs.Scene(
        #     show_viewer=self.args['vis'],
        # )
        self.scene = scene = gs.Scene(
            show_viewer=False,
        )

        plane = scene.add_entity(
            gs.morphs.Plane(),
        )

        self.table_pos = (0.78, -TABLE_WIDTH / 4, 0.02)
        self.table = scene.add_entity(
            material=gs.materials.Rigid(rho=5000),
                                        # friction=0.05),
                                        # coup_friction=0.05,),
            morph=gs.morphs.Box(
                size=(0.43, TABLE_WIDTH, TABLE_HEIGHT),
                pos=self.table_pos,
            )
        )

        self.cam_0 = scene.add_camera(
            pos=(2, 0, 1),
            lookat=(0.6, 0, 0.25),
            fov=40,
            GUI=False,
        )


        self.kinova = kinova = scene.add_entity(
            gs.morphs.URDF(
                file=str(pl.Path(__file__).parent / 'urdf/gen3_lite_2f_robotiq_85.urdf'),
                fixed=True,
                convexify=True,
                pos=(0.0, 0.0, 0.055), # raise to account for table mount
            ),
            material=gs.materials.Rigid(friction=1.0),
            vis_mode="collision"
        )


        self.teapot = teapot = scene.add_entity(
            gs.morphs.URDF(
                file=str(pl.Path(__file__).parent / 'urdf/teapot.urdf'),
                fixed=True,
                convexify=False,
                pos=TEAPOT_POSITION, # raise to account for table mount
            ),
            material=gs.materials.Rigid(friction=1.0),
            vis_mode="collision"
        )

        # self.mug = mug = scene.add_entity(
        #     gs.morphs.URDF(
        #         file=str(pl.Path(__file__).parent / 'urdf/mug.urdf'),
        #         fixed=True,
        #         convexify=False,
        #         pos=MUG_POSITION, # raise to account for table mount
        #     ),
        #     material=gs.materials.Rigid(friction=1.0),
        #     vis_mode="collision"
        # )
        self.mug = scene.add_entity(
            material=gs.materials.Rigid(rho=2500,
                                        friction=0.2),
            morph=gs.morphs.Cylinder(
                pos=MUG_POSITION,
                radius=0.0325,
                height=0.1,
            ),
)

        self.kdofs_idx = kdofs_idx = [kinova.get_joint(name).dof_idx_local for name in kinova_joint_names]
        eef = kinova.get_link(kinova_eef_name)
        # print(f"Kinova end effector: {eef}")
        scene.build()

        ############ Optional: set control gains ############
        # set positional gains
        kinova.set_dofs_kp(
            kp             = 3*np.array([100, 100, 100, 100, 100, 100, 100, 100, 100, 100]),
            dofs_idx_local = kdofs_idx,
        )
        kinova.set_dofs_position(np.array(KINOVA_START_DOFS_POS), kdofs_idx)

    def reset(self, trial_id=0, **kwargs):
        # Reset the scene and get the initial observation
        self.n_steps = 0

        if trial_id in TRIALS_POSITION_0:
            teapot_pos = POSITION_0
        elif trial_id in TRIALS_POSITION_1:
            teapot_pos = POSITION_1
        elif trial_id in TRIALS_POSITION_2:
            teapot_pos = POSITION_2
        else:
            rand_idx = random.randint(0,2)
            teapot_pos = [POSITION_0, POSITION_1, POSITION_2][rand_idx]

        self.mug.set_pos(MUG_POSITION); self.mug.set_quat(torch.Tensor([1, 0, 0, 0]))
        self.teapot.set_pos(teapot_pos); self.teapot.set_quat(torch.Tensor([1, 0, 0, 0]))
        
        self.table.set_pos(self.table_pos); self.table.set_quat(torch.Tensor([1, 0, 0, 0]))
        self.kinova.set_dofs_position(np.array(KINOVA_START_DOFS_POS), self.kdofs_idx)
        self.scene.step()
        obs = self.get_obs()
        if self.use_truncated_in_return:
            ret = obs, {}
        else:
            ret = obs
        return ret

    def compute_reward(self, obs):
        # go to point
        # goal_pos = MUG_POSITION
        # distance = torch.linalg.norm(mug_pos - orig_pos, ord=2, dim=-1, keepdim=True)
        # Push the mug
        # print(self.kinova.get_link("end_effector_link").get_pos())
        gripper_pos = self.kinova.get_link("end_effector_link").get_pos()
        distance = torch.linalg.norm(gripper_pos - MUG_POSITION, ord=2, dim=-1, keepdim=True)
        reward=-distance[0].item()
        # print(reward)
        # mug_pos = self.mug.get_pos()
        # orig_pos = MUG_POSITION
        # distance = torch.linalg.norm(mug_pos - orig_pos, ord=2, dim=-1, keepdim=True)
  
        # # reward for pushing farther away
        # if distance[0].item() > 0.2:
        #     reward = distance.item() 
        # else:
        #     reward = -1
        # reward for pushing farther away
        
        
        done = reward > -0.2
        if done: 
            self.successes.append(1)
            print(f"SUCCESS #", sum(self.successes))
            reward = 1.0
        else: 
            self.successes.append(0)

        return reward, done 