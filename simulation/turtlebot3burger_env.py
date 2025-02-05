'''
UNCORE - Team, 2025

This is a Gymnasium environment for our MuJoCo Turtlebot3 Burger model. 
The robot should learn to reach a goal (actually, a threshold around the goal) facing it with a certain angle.
The environment has not obstacles (the MJFC file includes some obstacles, but they are commented)

We have added some detailed comments on the parts of the code more tricky or sparsely documented elsewhere.

There are also a lot of print commands for debugging, most of them are commented.

# Based on:
# https://github.com/denisgriaznov/CustomMuJoCoEnviromentForRL
# https://gymnasium.farama.org/introduction/create_custom_env/
# https://www.gymlibrary.dev/content/basic_usage/
# https://safety-gymnasium.readthedocs.io/en/latest/introduction/about_safety_gymnasium.html
# https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html
'''

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import spaces
import gymnasium as gym
import os
from scipy.spatial import distance
import math
import glfw
import mujoco

np.seterr(invalid='raise')

# You can completely modify this class for your MuJoCo environment by following the directions.
# Pickle in Python is primarily used in serializing and deserializing a Python object structure. 
# In other words, it’s the process of converting a Python object into a byte stream to store it 
# in a file/database, maintain program state across sessions, or transport data over the network. 
# The pickled byte stream can be used to re-create the original object hierarchy 
# by unpickling the stream. This whole process is similar to object serialization in Java.
# When a byte stream is unpickled, the pickle module creates an instance 
# of the original object first and then populates the instance with the correct data. 
# This is generally needed only for environments which wrap C/C++ code, such as MuJoCo and Atari.
class Turtlebot3BurgerEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 1/(1000*0.001), 
        # Remember!! There is a relation between render_fps and property dt:
        # render_fps = dt, and dt = self.model.opt.timestep * self.frame_skip
        # where self.model.opt.timestep is the timestep value in the options tag of the MJFC model (s)
        # If this ratio is wrong, you get a AssertionError: Expected value: xxx, Actual value: yyy
        # frame_skip must be set in the MujocoEnv.__init__() call
        # The frame_skip default value is 5                
    }

    # set default episode_len for truncate episodes
    # **kwargs works just like *args, but instead of accepting positional arguments it accepts keyword (or named) arguments.
    def __init__(self, episode_len=1000, **kwargs):
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
        print("[IN ENV-INIT] Gymnasium version: ",gym.__version__)
        utils.EzPickle.__init__(self, **kwargs)
        # Define action and observation space
        # Both spaces must be gym.spaces objects   
        # In order to understand our observation space, please check the stateCode function at the bottom of this file 
        # Box low and high represent the lowest and highest values our observation space stores
        # If we use Discrete, we would need the functions to translate d-k to a integer value
        #observation_space = Box(low=0, high=64, shape=(8,), dtype=np.float32)
        observation_space = spaces.MultiBinary(6)
        #print("[IN ENV-INIT] Observation space defined")
        # Remember!! There is a relation between render_fps and property dt:
        # render_fps = dt, and dt = self.model.opt.timestep * self.frame_skip
        # where self.model.opt.timestep is the timestep value in the options tag of the MJFC model (s)
        # If this ratio is wrong, you get a AssertionError: Expected value: xxx, Actual value: yyy
        # The frame_skip default value is 5
        # Caution!! From this point, you can change self.frame_skip in this function,
        # and, since the metadata AssertionError is not checked again, your program will merrily keep on running...
        MujocoEnv.__init__(
            self,
            os.path.abspath("assets/turtlebot3burgermjmodel_scene.xml"),
            frame_skip=1000,
            observation_space=observation_space,
            **kwargs
        )
        # For now we are using a continuous space action with two values, one for each wheel
        # IMPORTANT: THE SPACE ACTION IS IMPLICITLY DEFINED FROM THE ACTUATORS DEFINITION IN THE MJFC MODEL
        # THE NUMBER OF CONTROL INPUTS DEPENDS ON THE NUMBER OF ACTUATORS IN THAT MODEL
        # AND THE CONTROL RANGE DEPENDS ON THE CTRLRANGE TAG OF SUCH ACTUATORS
        # IF YOU WANT TO OVERRIDE THAT DEFAULT SPACE ACTION, IT HAS TO BE DEFINED
        # AFTER THE MujocoEnv.__init__ FUNCTION
        self.action_space = spaces.Box(low=-1.0,high=1.0,shape=(2,),dtype="float32")
        #print("[IN ENV-INIT]:",self.action_space)
        self.step_number = 0
        self.episode_len = episode_len
        print("[IN ENV-INIT] Episode length: ",episode_len)
        print("[IN ENV-INIT] Init done!!")

    # Determine the reward depending on observation or other properties of the simulation
    # The action is chosen by the RL algorithm while training, 
    # and by the model.predict function in exploitation
    def step(self, action):
        # Remember!! Some important info about time values:
        #    - self.model.opt.timestep: this is the timestep value in the options tag of the MJFC model (s).
        #    - self.frame_skip: apparently, how many timesteps are made in between states
        #                      (I cannot find an official explanation of what this parameter means)
        #                      It is a parameter of the init function. Default value: 5.
        #    - self.dt: frameskip*timestep
        #               apparently, total duration of all substeps
        #    - Elapsed time in a do_simulation calling: frameskip*timestep
        #                                               i.e., the amount of time the action is performed (s)
        #                                               You should only provide the frameskip value,
        #                                               the final value frameskip*timestep is
        #                                               computed in the do_simulation function
        #    - metadata "render_fps" in the constructor: 1/dt
        #print("[IN ENV-STEP] Step action:", action)
        #print("[IN ENV-STEP] Step action time:", self.frame_skip*self.model.opt.timestep)
        #print("[IN ENV-STEP] dt:", self.dt)
        #print("[IN ENV-STEP] timestep:", self.model.opt.timestep)
        #print("[IN ENV-STEP] frameskip:", self.frame_skip)
        #print("[IN ENV-STEP] time before do_simulation:", self.data.time)
        #print("[IN ENV-STEP] Robot position:",self.init_qpos)
        #print("[IN ENV-STEP] Goal position:",self.data.geom('goal').xpos)
        #print("[IN ENV-STEP] GOAL:",self.data.geom('goal').xpos[0])
        self.do_simulation(action,self.frame_skip)
        #print("[IN ENV-STEP] time after do_simulation:", self.data.time)
        #input("Press Enter to continue...") 
        self.step_number += 1
        #print("[IN ENV-STEP] Step number:", self.step_number)
      
        obs = self._get_obs()
        #print("[IN ENV-STEP] observation after action:", obs)
        # The episode terminates if the robot reaches the terminal state,
        # that is, is quite near the goal and facing it with a certain orientation.
        terminated = bool((obs[0:6]==[0,0,0,0,0,0]).all())
        # We reward the robot only if it has reached the terminal state.
        # Furthermore, a get_reward function could be used
        # in case rewarding the robot even it does not reach the terminal state could be useful.
        if terminated :
           reward = 100
           #print("[IN ENV-STEP] Reward: ",reward) 
        else:
           #reward = self._get_reward()
           reward = 0
        # Truncated and terminated are not the same: https://farama.org/Gymnasium-Terminated-Truncated-Step-API
        # Since truncated is true only when the lenght of the episode is reached,
        # the agent does not know that this situation has arisen.
        # If we want that the agent gets some reward if the limit of the episode is reached,
        # (for example, in order to learn to reach the goal faster)
        # then truncated should be false, the reward set to a negative value when the episode length is exceeded, 
        # and terminated set to True.   
        truncated = self.step_number > self.episode_len
        return obs, reward, terminated, truncated, {}

    # Define what should happen when the model is reset (at the beginning of each episode)
    def reset_model(self):
        #print("[IN ENV-RESET] Step number:",self.step_number)
        self.step_number = 0
         
        # Some random noise is added to positions (and no to velocities, in our case)
        '''
        qpos = self.init_qpos + (self.np_random.uniform(
            size=self.model.nq, low=-2, high=2
        )+1)
        '''
        qpos = self.addNoise2qpos()
        self.set_state(qpos,self.init_qvel)
        
        # The goal position is also changed every time the RL algorithm is reset.
        # In order to keep those changes permanently,
        # we can read the position of the goal either from the model or from the data
        # but it must be written in the model (not in the data)
        oldGoalX = self.data.geom('goal').xpos[0]
        oldGoalY = self.data.geom('goal').xpos[1]
        #print("[IN ENV-RESET] Old goal pos:",self.model.geom('goal').pos)
        newGoalX = oldGoalX+np.random.uniform(-0.5,0.5)
        newGoalY = oldGoalY+np.random.uniform(-0.5,0.5)
        self.model.geom('goal').pos[0] = newGoalX
        self.model.geom('goal').pos[1] = newGoalY
        #print("[IN ENV-RESET] New goal pos:",self.model.geom('goal').pos)
        return self._get_obs()

    # Determine what should be added to the observation
    def _get_obs(self):
        robotMujocoPos=self.data.body('chassis').xpos
        robotMujocoQuat=self.data.body('chassis').xquat
        goalMujocoPos=self.data.geom('goal').xpos
        
        goal_robotDistance = distance.euclidean(robotMujocoPos,goalMujocoPos)
        goal_robotOrientation = math.atan2(goalMujocoPos[1]-robotMujocoPos[1],goalMujocoPos[0]-robotMujocoPos[0])
        goal_robotOrientation = goal_robotOrientation-(2*math.pi*math.floor(goal_robotOrientation/(2*math.pi)))
        
        qw = robotMujocoQuat[0]
        qx = robotMujocoQuat[1]
        qy = robotMujocoQuat[2]
        qz = robotMujocoQuat[3]
        robotYaw = math.atan2(2*(qw*qz+qx*qy),1-2*(pow(qy,2)+pow(qz,2)));
        goal_robotYaw = goal_robotOrientation-robotYaw
        goal_robotYaw = goal_robotYaw-(2*math.pi*math.floor(goal_robotYaw/(2*math.pi)))
        #print("[IN ENV-GETOBS] Distance:", robotDistance)
        #print("[IN ENV-GETOBS] Robot quaternion:", robotMujocoQuat)
        #print("[IN ENV-GETOBS] GoalX, RobX, GoalY, RobY:", goalMujocoPos[0],robotMujocoPos[0],goalMujocoPos[1],robotMujocoPos[1])
        #print("[IN ENV-GETOBS] Orientation rad:", goal_robotOrientation)
        #print("[IN ENV-GETOBS] Orientation deg:", math.degrees(goal_robotOrientation))
        #print("[IN ENV-GETOBS] Robot Yaw:", robotYaw)
        #print("[IN ENV-GETOBS] Robot Yaw deg:", math.degrees(robotYaw))
        #print("[IN ENV-GETOBS] Goal Robot Yaw:", goal_robotYaw)
        #print("[IN ENV-GETOBS] Goal Robot Yaw deg:", math.degrees(goal_robotYaw))
        obs = self.stateCode(goal_robotDistance,(goal_robotYaw))
        #print("[IN ENV-GETOBS] Observation:", obs)  
        # without this casting, envpy rises an AssertionError                              
        return obs.astype(np.int8)
        
   
    # Rewards the robot if the terminal state is not reached
    # It is not used at this moment.
    def _get_reward(self):
        robotPos=self.data.body('chassis').xpos
        goalPos=self.data.geom('goal').xpos
        robotDistance = distance.euclidean(robotPos,goalPos)
        reward = abs(1/robotDistance)
        if reward > 100 :
           reward = 100
        #print("[IN ENV-REWARD] Distance:", robotDistance)
        #print("[IN ENV-REWARD] Reward:", reward)                               
        return reward     
   
   # ----------------
   # AUXILIAR METHODS
   # ----------------
  
   # Adds a random uniform noise to the x,y coordinates of the chassis pose
    def addNoise2qpos(self):
        qposWithNoise = self.init_qpos
        qposWithNoise[0] = qposWithNoise[0]+np.random.uniform(-0.5,0.5)
        qposWithNoise[1] = qposWithNoise[1]+np.random.uniform(-0.5,0.5)
        return qposWithNoise

    # Translates the distance and orientation of the robot to the goal to a state of our observation space.
    # Our states are coded with binary six positions array:
    # positions 0-3 are for distance, positions 3-6 are for orientation.
    # Therefore, there are 8 possible values for distance, and the same for orientation.
    # Those values are determined depending on two thresholds: 
    # minRad for distance, which is the distance to the goal plus a safety radius,
    # diffAngle, which is the angle the robot should face the goal when it reaches it.
    # Hence, the terminal state is [0 0 0 0 0 0]
    def stateCode(self,dist,orient):
        minRad = 0.25
        diffAngle = 45
        
        if 0 <= dist < minRad:
           distCode = [0, 0, 0]
        elif minRad <= dist < 2*minRad:
           distCode = [0, 0, 1]
        elif 2*minRad <= dist < 3*minRad:
           distCode = [0, 1, 0]  
        elif 3*minRad <= dist < 4*minRad:
           distCode = [0, 1, 1]   
        elif 4*minRad <= dist < 5*minRad:
           distCode = [1, 0, 0]  
        elif 5*minRad <= dist < 6*minRad:
           distCode = [1, 0, 1]  
        elif 6*minRad <= dist < 7*minRad:
           distCode = [1, 1, 0]          
        else:
           distCode = [1, 1, 1]
           
        if 0 <= math.degrees(orient) < diffAngle:
           orientCode = [0, 0, 0]
        elif diffAngle <= math.degrees(orient) < 2*diffAngle:
           orientCode = [0, 0, 1]
        elif 2*diffAngle <= math.degrees(orient) < 3*diffAngle:
           orientCode = [0, 1, 0]  
        elif 3*diffAngle <= math.degrees(orient) < 4*diffAngle:
           orientCode = [0, 1, 1]   
        elif 4*diffAngle <= math.degrees(orient) < 5*diffAngle:
           orientCode = [1, 0, 0]  
        elif 5*diffAngle <= math.degrees(orient) < 6*diffAngle:
           orientCode = [1, 0, 1]  
        elif 6*diffAngle <= math.degrees(orient) < 7*diffAngle:
           orientCode = [1, 1, 0]          
        else:
           orientCode = [1, 1, 1]
                                                   
        return np.concatenate((distCode,orientCode))

    

