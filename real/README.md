# Turtlebot3 Burger basic movement package

This folder contains a basic movement package (tyrell_tb3_move) along with some of the messages required to perform that basic movement (tyrell_tb3_interfaces).

The basic movement package works like this:
- It listens to a Tyrell defined topic (tb3_speed_time) in order to get linear and angular speed commands, as well as the time they should be applied.
Then, it sends the speed commands to the turtlebot via the cmd_vel topic.
- It listens to the turtlebot /scan topic to get the lidar measurements and runs a (now) silly processing on those data.
Then, it sends the result of the processing to the Tyrell defined topic tb3_scan_sector

[DISCLAIMER] This is a WIP version and it is not fully tested.



