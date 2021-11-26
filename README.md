# Rapidly-exploring Random Tree

![alt text](https://github.com/lesurJ/RRT/blob/main/RRT.png)

## Introduction

In this repository, I propose a python implementation of the Rapidly-exploring Random Tree.

The development of this code was influenced by the following articles:
* the original paper by Steven M. Lavalle named *Rapidly-Exploring Random Trees: A New Tool for Path Planning.*
* the second paper by J.J. Kuffner and S.M. LaValle. *RRT-connect: An efficient approach to single-query path planning*
* [the rrt article on wikipedia](https://en.wikipedia.org/wiki/Rapidly-exploring_random_tree)

### Features

This implementation features both the single and the dual RRT search. The results are nicely plotted with mathplotlib.
Moreover, this RRT program has a "crude" path shortening function (named optimizedPath) and a smoothing function (smoothPath).
The latter making use of the Bézier curve in order to find a infinitely differentiable path !

### Output

The RRT computations are presented below for problems of dimension 2 and 3, respectively.

![alt text](https://github.com/lesurJ/RRT/blob/main/RRT_2.png)

![alt text](https://github.com/lesurJ/RRT/blob/main/RRT_3.png)


## How to use it

* In an "out of the box" fashion

```python
if __name__=='__main__':
    # 1. Choose dimension of the problem.
    N = 2
    # 2. Instantiate the RRT object with N, you may indicate the start and the goal configurations.
    r = RRT(N, start=-0.75*np.ones(N), goal=0.75*np.ones(N))
    # 3. Run the RRT planner (the default one uses bidirectional search)
    r.runRRT(mode='dual')
    # 4. Plot the results
    r.plot()

```

* In a more sophisticated fasion

You'll have to set the collision detection function according to your need. In this implementation, I am using a random obstacle generator and for detecting a collision, the program loops over all obstacles. A faster and more reliable implementation of the collision detection can be achieved using a simulation like Pybullet.

The path processing functions (optimizePath, smoothPath) are not optimal yet and should definitely be improved. 

## Remarks

I strongly recommend you to take a look at the papers cited above to get an intuition of the algorithm.

As the output of RRT is very chaotic due to the random sampling in the search space, one needs to post-process it. In this implementation, I am using a custom function to remove unnecessary points while taking care of the obstacle avoidance requirement. For a better result, one might want to make use of the Shortcutting algorithm (see [osrobotics](https://www.osrobotics.org/osr/planning/post_processing.html)). I also implemented a path-smoothing function using a Bézier curve of order n where n is the number of points in the path.

All of this works well for high dimensional problems. However, the plotting function won't work anymore :)

*Note : This planner can easily be used with robotic manipulators. I successfully used it on a KUKA iiwa7 robot. For this, the RRT was conducted in the joint space of the robot while the collision detection was done in the task space. I used the PyBullet simulator with an urdf file of my robot. The plotting function was slightly modified to project the Tool-Center-Point of the robot with the help of the forward kinematics.*

