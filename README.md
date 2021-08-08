# RRT
Rapidly-exploring Random Tree

## Introduction

In this repository, I propose a python implementation of the Rapidly-exploring Random Tree.

For the development of this code I was inspired by
* the original paper by Steven M. Lavalle named *Rapidly-Exploring Random Trees: A New Tool for Path Planning.*
* the second paper by J.J. Kuffner and S.M. LaValle. *RRT-connect: An efficient approach to single-query path planning*
* [the rrt article on wikipedia](https://en.wikipedia.org/wiki/Rapidly-exploring_random_tree)

## Remarks

I assume that you are already familiar with the concept of RRT.

The hyperparameters must be set according to your needs.

The collision detection function must be set according to your needs. A geometrical approach (based on obstacles shapes) works well but requires a loop over all obstacles and can become cumbersome with complex obstacles. The use of a simulation taking care of collision is therefore preferred (e.g Pybullet as it is very easy to use and fast!).

By default, this implementation uses a bidirectionnal search. 




## How To Use