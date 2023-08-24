# Rapidly-Exploring Random Tree (RRT)

The first version of RRT has been implemented and tested.

Input: grid map (.txt), start point, goal, epsilon (step size), robot radius

Output: a tree graph and a path from start to goal node.

## Feature Summary:

Have's:

* Random sample generator
* Goal-unbiased sampling: currently no bias/heuristics is added
* Collision check
* Animation

Doesn't Have's:

* Goal-biased sampling
* Path smoothing
* Generic grid map: currently square grid map is assumed.

## Future Plan:

* Add those currently unavailable features
* Data structure optimization

## Dependencies:

* numpy
* matplotlib
* math
* Occupancy_Grid_Mapping

## Quick Demo:

<img src="Successful_Run.png" width="50%" height="50%">

<img src="Failed_Run.png" width="50%" height="50%">

https://user-images.githubusercontent.com/63512723/235427834-49c4389c-5919-4861-952b-32e57501c95b.mp4
