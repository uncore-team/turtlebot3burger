# Turtlebot3 Burger simulation with MuJoCo, Gymnasium and Stable Baselines3

*** This is a WIP project. It is functional and profusely documented, though it has not been thorougly tested ***

This folder contains several files that allow to create a MuJoCo custom environment for Turtlebot3 Burger that can be used with Gymnasium in order to learn some task using RL algorithms included in Stable Baselines 3:

- The MJFC MuJoCo model for Turtlebot3 Burger (turtlebot3burgermjmodel_diff.xml) and a MuJoCo xml scene that includes that model (urtlebot3burgermjmodel_scene.xml)
- A python script (tb3.py) that can be useful to understand the MJFC model and data structure, and how these elements can be accesed from python.
- Finally, three files that covers the whole process of training and testing a RL algorithm with the MuJoCo model using Gymnasium and Stable Baselines 3:
    - turtlebot3burger_env.py integrates the Turtlebot3 model into a Gymnasium environment ready to learn a task with RL.
    - turtlebot3burger_rl.py applies a Stable Baselines 3 RL algorithm to the previous Gymnasium environment.
    - turtlebot3burger_test.py tries and renders the model learnt with the previous RL algorithm, so you can visually check the goodness of the learning process.

    The workflow with this three python scripts would be the following:
    - First you modify (or create a new one) the turtleb3burger_env.py in order to adapt to your task (the task we have learnt in this example is reaching a goal in an unpopulated environment). You don't have to run this file.
    - Then you launch the turtlebot3burger_rl.py file (maybe you will have to do some changes to adapt it to your task). After running this file you will have a trained RL model. At this point you can also run TensorBoard and see significant parameters of the trained model (the mean reward, for example)
    - Finally you can run turtlebot2burger_test.py (again, with the modifications you need) with the model trained in the previous step, and see its performance in a visual environment.
