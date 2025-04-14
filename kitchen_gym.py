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

PX, PZ = 0.7, 0.17
POSITION_0 = torch.tensor((PX, -0.1, PZ))
POSITION_1 = torch.tensor((PX, -0.05, PZ))
POSITION_2 = torch.tensor((PX, -0.2, PZ))

## Default Args
DEFAULT_RADIUS = 0.034
DEFAULT_HEIGHT = 0.09
DEFAULT_RHO = 2000
DEFAULT_FRICTION = 0.5
DEFAULT_STARTING_X = 0.65 


class KitchenGym(GenesisGym):
    """
    Custom Gymnasium environment for the Genesis game.
    """
    def __init__(self, args={}, size=(96, 96), use_truncated_in_return=False):
        super().__init__(args)

    def init_env(self):
        self.kp = kp = 5

        self.scene = scene = gs.Scene(
            show_viewer=self.args['vis'],
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
            GUI=True,
        )

        import pathlib as pl

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

        self.mug = mug = scene.add_entity(
            gs.morphs.URDF(
                file=str(pl.Path(__file__).parent / 'urdf/mug.urdf'),
                fixed=True,
                convexify=False,
                pos=MUG_POSITION, # raise to account for table mount
            ),
            material=gs.materials.Rigid(friction=1.0),
            vis_mode="collision"
        )

        from kinova import JOINT_NAMES as kinova_joint_names, EEF_NAME as kinova_eef_name, TRIALS_POSITION_0, TRIALS_POSITION_1, TRIALS_POSITION_2
        self.kdofs_idx = kdofs_idx = [kinova.get_joint(name).dof_idx_local for name in kinova_joint_names]
        eef = kinova.get_link(kinova_eef_name)
        print(f"Kinova end effector: {eef}")
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
        teapot_pos = self.teapot.get_pos()
        goal_pos = self.mug.get_pos()
        distance = torch.linalg.norm(teapot_pos - goal_pos, ord=2, dim=-1, keepdim=True)

        reward = -distance.item() # TODO: implement reward function
        done = reward > -0.1 and (teapot_pos[2].cpu().numpy().item() >= (MUG_POSITION[2] - 0.07)) and (goal_pos[2].cpu().numpy().item() >= (MUG_POSITION[2] - 0.07))
        if done: 
            print(f"SUCCESS!")
            reward = 1.0
        elif self.force_sparse:
            reward = 0.0

        return reward, done # TODO: implement reward function

    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Genesis Gym Environment')
    parser.add_argument('--vis', action='store_true', help='Enable visualization')
    parser.add_argument('--radius', type=float, default=DEFAULT_RADIUS, help='Bottle radius')
    parser.add_argument('-e', '--height', type=float, default=DEFAULT_HEIGHT, help='Bottle height')
    parser.add_argument('-o', '--rho', type=float, default=DEFAULT_RHO, help='Density of the bottle')
    parser.add_argument('--friction', type=float, default=DEFAULT_FRICTION, help='Friction of the bottle')
    parser.add_argument('--starting_x', type=float, default=DEFAULT_STARTING_X, help='Starting x position of the bottle')
    parser.add_argument('--max-demos', type=int, default=1e7, help='Max number of demos to load')
    args = parser.parse_args()
    

    env = KitchenGym(args)
    obs = env.reset()
    done = False
    max_reward = float('-inf')
    trials = 1; successful_trials = 0; steps = 0; pickups = 0


    while True:
        action = {'action': env.action_space.sample()}  # Sample random action
        # action = demo_player.next_action(normalize=False)
        print("action: " , action)
        if action is None or steps > env._max_episode_steps() or done:
            bottleZ = env.mug.get_pos().cpu().numpy()[2]
            print(f"\t Max Reward {max_reward:+1.2f}. {bottleZ=}")
            max_reward = float('-inf')
            # trial_id = demo_player.next_demo()
            if done: successful_trials += 1
            if bottleZ > 0.15: pickups += 1
            # if trial_id == -1:
            #     print("No more demos")
            #     break
            trials += 1; steps = 0; done = False
            # env.reset(trial_id=trial_id)
            env.reset()
        else:
            steps += 1
            obs, reward, done, *_ = env.step(action['action'])
            if args.vis: env.render(use_imshow=True)
            if reward > max_reward:
                max_reward = reward
            
            # if reward > -0.10:
            #     print(f"Reward: {reward}")

    print(f"Trials: {trials} Successful Trials: {successful_trials} Success Rate: {successful_trials/trials:.2%}")
    print(f"Pickups: {pickups} Pickup Rate: {pickups/trials:.2%}")


    # action structure:
    # action:  {'action': array([ 0.41947876, -1.5054056 ,  1.70436717, -1.3545354 ,  1.62811608,-1.63153025,  0.50537103])}
