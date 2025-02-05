'''
UNCORE - Team, 2025

This is python script that helps to understand how the MuJoCo MJFC model of our Turtlebot3 Burger
can be read from python.

We have added some detailed comments on the parts of the code that are more tricky or sparsely documented elsewhere.

There are also a lot of print commands for debugging, most of them are commented.

Based on:
   · How to move the robot: https://mujoco.readthedocs.io/en/2.3.6/python.html
   · How the actuators control works: https://www.roboti.us/forum/index.php?threads/directly-asign-joint-angles.3353/
'''



import mujoco
from mujoco.glfw import glfw
from mujoco import viewer
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
import time
import itertools
import mediapy as media
import matplotlib.pyplot as plt
from loop_rate_limiters import RateLimiter
import math


# ----------------------
# POSE RELATED FUNCTIONS
# ----------------------

# Info about quaternions:
# https://mujoco.readthedocs.io/en/2.2.1/programming.html
# https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles

# Returns the Euler angle in radians from the quaternion (rotation around z axis)
def quat2angle(w,qx,qy,qz):
  sinycosp = 2*(w*qz+qx*qy)
  cosycosp = 1-2*(qy*qy+qz*qz)
  angle = math.atan2(sinycosp,cosycosp)
  return angle
  
# Returns the pose from the xyz position values and a quaternion (rotation around z axis)
def getPose(pq):
  return [pq[0],pq[1],quat2angle(pq[3],pq[4],pq[5],pq[6])]
 
 
# -----------------------------------------------------------------
# GETTERS OF THE POSITIONS, VELOCITIES AND SENSOR DATA OF THE ROBOT
# -----------------------------------------------------------------
  
# Returns the qpos of the chassis
# That is, a vector of length 7 with the xyz position and a quaternion for rotation around z axis
def getChassisqpos(robotqpos):
  return robotqpos[0:7]
  
# Returns the qpos of the wheels
# That is, a vector of length 2 with the angles of each wheel (left,right)
def getWheelsqpos(robotqpos):
  return [robotqpos[11],robotqpos[12]]
  
# Returns the sensor readings of the wheels
# That is, a vector of length 2 with the velocity readings of each wheel (left,right)
def getWheelsSensors(robotsensordata):
  return [robotsensordata[0],robotsensordata[1]]
  
# Returns the readings of the complete FoV of the 300º lidar defined in the XML model
# Since some other sensors are defined in the XML model before the lidar, 
# an offset is added to the 360 lidar angles.
def getLidar360(robotsensordata):
  return robotsensordata[8:368]
  
# Returns a specific readings of the 300º lidar defined in the XML model
# Since some other sensors are defined in the XML model before the lidar, 
# an offset is added to the 360 lidar angles.
def getLidarReading(robotsensordata,lidarAngle):
  lidarOffset = 8
  return [robotsensordata[lidarAngle+lidarOffset]]
  
# Returns the readings of a cone fromAngle toAngle of the 300º lidar defined in the XML model
# Since some other sensors are defined in the XML model before the lidar, 
# an offset is added to the 360 lidar angles.
def getLidarCone(robotsensordata,fromAngle,toAngle):
  lidarOffset = 8
  return robotsensordata[fromAngle+lidarOffset:toAngle+lidarOffset]


# ----------
# SIMULATION
# ----------

# Make model and data
model = mujoco.MjModel.from_xml_path('assets/turtlebot3burgermjmodel_scene.xml')
data = mujoco.MjData(model)

# Degrees of freedom: 11
#    - free joint (chassis) has 6 DoF
#    - ball joint (caster wheel) has 3DoF
#    - hinge joint (wheel) has 1 DoF, therefore there are 2 DoFs
print('Total number of DoFs in the model:', model.nv) # 
# Generalized positions: 13
#    - free joint (chassis) has 3 values xyz and a quaternion = 7
#    - ball joint (caster wheel) has a quaternion = 4
#    - hinge joints (wheels) are angles = 2
print('Generalized positions:', data.qpos) 
# Generalized positions: 11
#    - free joint: 3 linear, 3 angular = 6
#    - ball joint (caster wheel): 3 angular
#    - hinge joints (wheels): 2 (I guess angular)
print('Generalized velocities:', data.qvel) 
print('Control vector: ',data.ctrl)

# enable joint visualization option:
scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
scene_option.frame = mujoco.mjtFrame.mjFRAME_GEOM

duration = 20  # (seconds)
framerate = 60  # (Hz)

# Simulate and display video.
# Launch viewer
# We choose the passive viewer, so the rest of the code is executed
# However, the mouse is not working on the viewer (unless it is synchronized)
# and the user must control the time and advance of the simulation
with mujoco.viewer.launch_passive(model, data) as viewer:
  start = time.time()
  # The speed of each wheel should be set out of the loop, with zero value.
  # Once the speed value is initialized, it should not be modified directly in the loop:
  # the speed will change accordingly to the speed reference we set in each step of the loop.
  # If there is no control command in the loop the wheel speed decreases until it stops.
  data.qvel[9] = 0
  data.qvel[10] = 0
  while viewer.is_running() and time.time() - start < duration:
    step_start = time.time() 
    print('\n[NEW SIMULATION STEP]')
    # The positions and orientations of the body/geom/site frames are computed at each time step
    # from mjData.qpos via forward kinematics. 
    # The results of forward kinematics are available in mjData as xpos  
    pos_chassis = data.body('chassis').xpos
    print('Chassis position: ',pos_chassis)
    chassisqpos = getChassisqpos(data.qpos)
    print('Chassis pose: ',getPose(chassisqpos))
    # The control reference seems to be a speed reference
    # Not sure yet if it is linear or angular speed, and which units...
    # The control vector has a position for each actuator defined in the XML model
    data.ctrl[0] = 0.15;
    data.ctrl[1] = -0.15;
    # data.qpos stores the angles of the joints. The wheel joints are the last two values
    # apparently the ouput is in radians, despite the mujoco model compiler is set in degrees
    # because all angular quantities mjModel and mjData are expressed in radians.
    print('Wheels angles (L R): ',getWheelsqpos(data.qpos))
    # Sensor readings are stored in the data.sensordata structure
    # The fields of that structure are defined according to the order of the sensors in the XML model
    # The velocity sensor returns the same value stored in the data.qvel variable
    print('Wheels velocities (L R): ',getWheelsSensors(data.sensordata))
    # The velocimeter is mounted at a site, and has the same position and orientation as the site frame
    # It returns the 3 values of the linear velocity of the site in local coordinates
    #print('Left velocimeter:', data.sensordata[2],data.sensordata[3],data.sensordata[4])
    #print('Right velocimeter:', data.sensordata[5],data.sensordata[6],data.sensordata[7])
    # The lidar defined in the XML file has a 360 FoV. Since it is the fifth sensor,
    # you need to add an offset with value = 4 if you access directly the data.sensordata reading.
    # However, the functions defined in this script encapsulates that offset
    # so you can use the 0-359 values for accesing precise positions or cones.
    # lidar returns -1 if does not detect anything
    #print('Lidar 0:', data.sensordata[363])
    #print('Lidar 1:', data.sensordata[4])
    #print('Tosjuntos: ',data.sensordata) 

    mujoco.mj_step(model, data)
    viewer.sync()
 
    time_until_next_step = model.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)
