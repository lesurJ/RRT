# Rapidly-exploring Random Tree

![alt text](https://github.com/lesurJ/RRT/blob/main/RRT.png)

## Introduction

In this repository, I propose a python implementation of the Rapidly-exploring Random Tree.

The development of this code was influenced by the following articles:
* the original paper by Steven M. Lavalle named *Rapidly-Exploring Random Trees: A New Tool for Path Planning.*
* the second paper by J.J. Kuffner and S.M. LaValle. *RRT-connect: An efficient approach to single-query path planning*
* [the rrt article on wikipedia](https://en.wikipedia.org/wiki/Rapidly-exploring_random_tree)

## Remarks

I assume that you are already familiar with the concept of RRT.

The hyperparameters must be set according to your needs. By default, the stepsize alpha is at 0.1 and the bias beta is at 0.5. I implemented two versions for the RRT : the unidirectional and bidirectional search; the latter being the default one as it is more robust in constrained environments.

The collision detection function (**checkCollision, line 242**) must be set according to your needs. 
A geometrical approach (based on obstacles shapes) works well but requires a loop over all obstacles and can become cumbersome with complex obstacles. In the actual implementation, the RRT tries to avoid a circle with radius 0.25 centered at the origin. 
A faster implementation can be achieved with a simulation taking care of the collision detection (e.g Pybullet as it is very easy to use and fast!).

As the output of RRT is very chaotic due to the random sampling in the search space, one needs to post-process it. In this implementation, I am using a custom function to remove unnecessary points while taking care of the obstacle avoidance requirement. For a better result, one might want to make use of the Shortcutting algorithm (see [osrobotics](https://www.osrobotics.org/osr/planning/post_processing.html)) 

All of this works well for high dimensional problems. However, the plotting function won't work anymore :)

*Note : This planner can easily be used with robotic manipulators. I successfully used it on a KUKA iiwa7 robot. For this, the RRT was conducted in the joint space of the robot while the collision detection was done in the task space. I used the PyBullet simulator with an urdf file of my robot. The plotting function was slightly modified to project the Tool-Center-Point of the robot with the help of the forward kinematic.*

## How To Use
