'''
UNCORE - Team, 2025

This is a Stable Baselines3 for the Turtlebot 3 Burger MuJoCo environment created in the turtlebot2burger_env.py file.
To learn more about this, check:
https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
https://colab.research.google.com/github/araffin/rl-tutorial-jnrr19/blob/sb3/5_custom_gym_env.ipynb

Among all the different RL algorithms provided by Stable Baselines3, you have to pick one. 
When it comes to choose a RL algorithm, you should check before which type of observations and actions spaces
can handle (continuous, discrete...). Remember that the observations and actions spaces are defined
in the gymnasium file that creates your environment (turtlebot3burger_env.py, in our case).

We have used SAC algorithm (although you will find also the commented code for running the A2C algorithm).
Depending on the computing power of your computer, you can use vectorized environments;
we have prepared both scenarios, though the non vectorized code is commented.

It also stores TensorBoard information that helps to analyse the performance of the reinforcement learning process
after running your experiments.

We have added some detailed comments on the parts of the code that are more tricky or sparsely documented elsewhere, as well as some time measurement info.

There are also a lot of print commands for debugging, most of them are commented.

Time control explained in:
https://pynative.com/python-get-execution-time-of-program/#example-get-program-s-execution-time-in-seconds

Logger callbacks explained in:
https://stackoverflow.com/questions/69181347/stable-baselines3-log-rewards
https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html
'''

import stable_baselines3
from stable_baselines3.common.env_checker import check_env
from turtlebot3burger_env import Turtlebot3BurgerEnv
from stable_baselines3 import SAC,PPO, A2C
from stable_baselines3.sac import MlpPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam
from stable_baselines3.common.logger import configure
import glfw
import time

class TensorboardCallback(BaseCallback):
    """
    Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)

    # Apparently, it logs in TensorBoard the values you define in _on_step()
    # Up to this moment, I'm only able to record the actions
    # (TensorBoard logs the rewards by default)
    # If you are using vectorized environments in your experiment
    # and you want to log the actions of an specific environment,
    # you should add the [index] of the environment after the self.locals array
    # The actions are displayed as an histogram in TensorBoard,
    # however, it is not clear how to read that histogram in this problem, 
    # where the action is an array with two values.
    def _on_step(self) -> bool:
        #r = self.locals['rewards'][0]
        #self.logger.record('reward', r)
        a = self.locals['actions']
        self.logger.record('action', a)
        #print('[IN RL-TSBOARD] action: ',a)
        #o = self.locals['observations'][0]
        #self.logger.record('obs', o)
        return True

# Start time
st = time.time()
# Initialize your enviroment
env = Turtlebot3BurgerEnv(render_mode="human",width=1200,height=600)

# It will check your custom environment and output additional warnings if needed
print("[IN RL] I'm going to check the environment...")
check_env(env)
print("[IN RL] Environment checked!!")
# The reset method is called at the beginning of an episode
# Anyway, since we are using vectorized environments, this is not necessary:
# the vectorized environments reset themselves automatically according to:
# https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html
print("[IN RL] I'm going to reset the environment...")
obs, info = env.reset()
print("[IN RL] Environment reset!!")

# Mujoco 3.0.0 and gymnasium 0.29 crashes when human render mode is set
# https://github.com/Farama-Foundation/Gymnasium/issues/749
# In order to solve this, gymnasium 1.0.0 and stable baselines3 2.5 are required
# However, pip install does not install that stable baselines3 version,
# so you have to install the bleeding-edge version:
# https://stable-baselines3.readthedocs.io/en/master/guide/install.html
# This solves the main error, but still a Exception ignored arises, though
# Exception ignored in: <function WindowViewer.__del__ at 0x78d5eefc89d0>
# AttributeError: 'NoneType' object has no attribute 'glfwGetCurrentContext'
# It seems that the integration between Mujoco 3.2.6 and gymnasium 1.0.0 
# is not seamless yet:
# https://stackoverflow.com/questions/75157791/attributeerror-nonetype-object-has-no-attribute-glfwgetcurrentcontext
print("[IN RL] Stable baselines3 version: ",stable_baselines3.__version__)
print("[IN RL] Observation space:", env.observation_space)
print("[IN RL] Shape:", env.observation_space.shape)
print("[IN RL] Action space:", env.action_space)

#----
# A2C
#----

# vectorized envs creating
# According to https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html,
# "A2C is meant to be run primarily on the CPU, especially when you are not using a CNN. 
# To improve CPU utilization, try turning off the GPU and using SubprocVecEnv instead of the default DummyVecEnv."
'''
if __name__=="__main__":
   # Check how to use vec envs before use it: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html
   #vec_env = make_vec_env(Turtlebot3BurgerEnv, n_envs=4, vec_env_cls=SubprocVecEnv)
   #model = A2C("MlpPolicy", vec_env, device="cpu", tensorboard_log="logs/sac/test",verbose=1)
   model = A2C("MlpPolicy", env, tensorboard_log="logs/tests",verbose=1)
   print("[IN RL] I'm about to learn...")
   model.learn(total_timesteps=50000, log_interval=100, tb_log_name="a2c-1-1_50000", callback=TensorboardCallback())
   print("[IN RL] I've learnt :)")
   model.save("a2c_tb3burger_reachgoal")
   env.close()
   #vec_env.close()
'''

#----
# SAC
#----
# If log_interval*episode_len < total_timesteps, it does not record any data for TensorBoard,
# and there is no rollout either.

'''
# Non vectorized environment
env = Monitor(env)
model = SAC("MlpPolicy", env, tensorboard_log="logs/tests",verbose=1)
print("[IN RL] I'm about to learn...")
model.learn(total_timesteps=100000, log_interval=1, tb_log_name="sac_100000_fs1000_binrw", callback=TensorboardCallback())
print("[IN RL] I've learnt :)")
model.save("sac_tb3burger_reachgoal")
env.close()
'''

# Vectorized environment
# Depending on the computing power of your computer, you can use more vectorized environments changing n_envs value
# The SAC algorithm needs a lot of episodes for learning properly. Hence the high value of total_timesteps
if __name__=="__main__":
   vec_env = make_vec_env(Turtlebot3BurgerEnv, n_envs=1, vec_env_cls=SubprocVecEnv)
   model = SAC("MlpPolicy", vec_env,device="cuda",tensorboard_log="logs/tests",verbose=1)
   print("[IN RL] I'm about to learn...")
   model.learn(total_timesteps=100000, log_interval=1, tb_log_name="experiment", callback=TensorboardCallback())
   print("[IN RL] I've learnt :)")
   model.save("sac_tb3burger_reachgoal")
   vec_env.close()


# End time
et = time.time()

# Elapsed time
elapsed_time = et - st
print('[IN RL] Elapsed time:', elapsed_time/60, 'minutes')
