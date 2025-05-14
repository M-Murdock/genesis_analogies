# Analogical Explanations

### Training Kitchen Policy
Run ``train_kitchen_gym.py`` to train and save the policy
    - Specify the max steps/episodes
    - Specify the filename for saving the policy

Run ``load_kitchen_sim.py`` to load the trained policy 

### Environments
``KitchenGym`` in ``kitchen_gym.py`` extends ``GenesisGym``
- Initializes kitchen environment
- Specifies locations of objects
- Creates function for resetting environment
- Specifies reward function

``GenesisGym`` in ``genesis_gym.py`` is a custom genesis environment. 
- Contains basic functions for getting observations and gripper force, rendering the scene, and applying an action

### Misc
``kinova.py`` specifies the robot's joint names and positions