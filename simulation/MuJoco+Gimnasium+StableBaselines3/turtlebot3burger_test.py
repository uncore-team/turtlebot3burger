'''
UNCORE - Team, 2025

This file launches and renders a set of episodes using a previously trained RL model, so you can visually check
the performance of your model.
Apart from the rendering, we have also added some auxiliary functions that prints on screen
our states (aka observations), so we can analyse, for every episode, some important information 
(terminated/truncated, length of the episode, actions...)

Please notice that the state related functions are designed for the observation space of our task, which is reaching a goal in an environment with no obstacles, and whose observation space is explained in turtlebot3burger_env.py
If your observation space is different, you should recode those functions.

We have added some detailed comments on the parts of the code that are more tricky or sparsely documented elsewhere.

There are also a lot of print commands for debugging, most of them are commented.

Based on:
https://github.com/denisgriaznov/CustomMuJoCoEnviromentForRL
'''

from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from stable_baselines3 import SAC, PPO, A2C
from turtlebot3burger_env import Turtlebot3BurgerEnv
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# -------------------
# AUXILIARY FUNCTIONS
# -------------------

def getStateIndex(state):
   print("[IN TEST-getStateIndex] State:", state)
   distIndex = array2Index(state[0:3])
   orientIndex = array2Index(state[3:6])
   stateIndex = (distIndex*10)+orientIndex  
   return stateIndex
   
def array2Index(substate):
   substateArray = substate.tolist()
   match substateArray:
    case [0, 0, 0]:
        index = 0
    case [0, 0, 1]:
        index = 1
    case [0, 1, 0]:
        index = 2
    case [0, 1, 1]:
        index = 3
    case [1, 0, 0]:
        index = 4
    case [1, 0, 1]:
        index = 5
    case [1, 1, 0]:
        index = 6
    case [1, 1, 1]:
        index = 7    
    case _:
        raise ValueError("Wrong Substate!!")       
   return index

def index2BinArray(index):
   match index:
     case 0:
        binArray = [0, 0, 0]
     case 1:
        binArray = [0, 0, 1]
     case 2:
        binArray = [0, 1, 0]
     case 3:
        binArray = [0, 1, 1]
     case 4:
        binArray = [1, 0, 0]
     case 5:
        binArray = [1, 0, 1]
     case 6:
        binArray = [1, 1, 0]
     case 7:
        binArray = [1, 1, 1]    
     case _:
        raise ValueError("Wrong code!!")
   return binArray

def getStateCode(stateIndex):
    distCode, orientCode = divmod(stateIndex,10)
    #print("[IN TEST-getStateCode] State Index:", stateIndex)
    #print("[IN TEST-getStateCode] Distance Code:", distCode," Orientation Code:", orientCode)
    distArray = index2BinArray(distCode)
    orientArray = index2BinArray(orientCode)
    #print("[IN TEST-getStateCode] Distance Array:", distArray," Orientation Array:", orientArray)
    return np.concatenate((distArray,orientArray))
    
    
# ---------
# MAIN CODE
# ---------

# CAUTION!! Remember that the Mujoco model (.xml file) and learning environment (*_env.py file) 
# called by this test script must be the same files you used for training your model.
# Furthermore, also check that the .zip trained model you load in this script is 
# the same file you got after training your model.

env = Turtlebot3BurgerEnv(render_mode="human",width=1200,height=600)
model = SAC.load("sac_1000000_fs1000_noobs_binrw_or_chgoal_4.zip")

num_eval_episodes = 500
buffer_length=num_eval_episodes

'''
Uncomment in case you want to use the video recording wrapper. The render_mode should be "rgb_array"
env = RecordVideo(env, video_folder="VID_tb3_reachgoal_ppo", name_prefix="eval",
                  episode_trigger=lambda x: True)
'''
#env = RecordEpisodeStatistics(env, buffer_length)

dictStatesActions = defaultdict(list)

for episode_num in range(num_eval_episodes):
   episode_over = False
   numSteps = 0
   obs, info = env.reset()
   while not episode_over :
      print("[IN TEST] Episode:", episode_num)
      print("[IN TEST] Num steps:", numSteps)
      action, _states = model.predict(obs)
      #print("[IN TEST] Action:", action)
      #print("[IN TEST] Observation before action:", obs)
      stateIndex = getStateIndex(obs)
      print("[IN TEST] State index:", stateIndex)
      obs, reward, terminated, truncated, info = env.step(action)
      dictStatesActions[stateIndex].append(action)
      episode_over = terminated or truncated
      numSteps = numSteps+1
      print("[IN TEST] Observation after action:", obs)
      print("[IN TEST] Terminated:", terminated, " Truncated:", truncated)
      env.render()
env.close()


# Scatterplot that maps states to actions


#print("[IN TEST] Dictionary:", dictStatesActions)

# Our states are six valued binary arrays: position is stored in indexes 0:3, and orientation in indexes 3:6
# In order to store the states in the dictionary, we split the binary array in two subarrays
# and then translate each of them to binary. For example, state [0 1 1 1 0 0] turns to 37.

row = 8
col = 8

for dictIndex in dictStatesActions:
        actionsList = dictStatesActions[dictIndex]
        #print("[IN TEST] Dictionary Index:", dictIndex)  
        #print("[IN TEST] ActionsList:", actionsList)
        xList = []
        yList = []
        for numAction in range(len(actionsList)):
           #print("[IN TEST] Num. Action:", numAction)
           #print("[IN TEST] Action:", actionsList[numAction])
           #print("[IN TEST] Action X:", actionsList[numAction][0])  
           #print("[IN TEST] Action Y:", actionsList[numAction][1])
           xList.append(actionsList[numAction][0])
           yList.append(actionsList[numAction][1])
        #print("[IN TEST-Scatter] Dictionary Index:", dictIndex)  
        # Subplot index must be in the row*col range, therefore we need to adapt
        # our state codification (non linear) to the linear one required by subplot
        # which is an octal value, since the range of our state codification for distance and orientation 
        # covers the range from 000 to 111    
        scatterIndex = int(str(dictIndex), 8)
        #print("[IN TEST-Scatter] Scatter Index:", scatterIndex) 
        ax = plt.subplot(row,col,scatterIndex+1)
        plt.scatter(xList,yList,c="darkcyan")
        plt.xlabel("L-wheel",fontsize=6)
        plt.ylabel("R-wheel",fontsize=6)
        plt.title('State '+np.array2string(getStateCode(dictIndex)),fontsize=6)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        plt.setp(ax.get_xticklabels(),fontsize=4)
        plt.setp(ax.get_yticklabels(),fontsize=4)
        plt.tight_layout()
plt.suptitle("STATES-ACTIONS MAP")
plt.show()


'''
obs, info = env.reset()
frames = []
for _ in range(500):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    image = env.render()
    if _ % 5 == 0:
        frames.append(image)
    cv2.imshow("image", image)
    cv2.waitKey(1)
    if done or truncated:
        obs, info = env.reset()

# uncomment to save result as gif
with imageio.get_writer("media/test.gif", mode="I") as writer:
    for idx, frame in enumerate(frames):
        writer.append_data(frame)
'''
