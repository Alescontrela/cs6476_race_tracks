## Problem Statement
To track [autorally](https://arxiv.org/pdf/1806.00678.pdf) vehicle pose (position & orientation) and twist (linear and angular velocity) in 3D of a leading vehicle from the on-board camera of a chasing vehicle. The system would be fed a sequence of images from a monocular camera, and the output would be a real-time estimate of the leading vehicle's pose and twist relative to the chasing vehicle. 


<p align="center">
  <a href="https://www.youtube.com/watch?v=FbcGs-XoiUw">
    <img src="https://img.youtube.com/vi/FbcGs-XoiUw/0.jpg"/>
  </a> <br/>
  Video demonstrating a single autorally vehicle operating around a track at verious velocities.
</p>

## Mission Statement
To track the 3D pose and twist of  a leading autorally vehicle from the onboard camera of a chasing vehicle. The input is a sequence of images which may or may not contain N leading vehicles. The output would be the 3D pose and twist of the N leading vehicles relative to the chasing vehicle. The purpose of the detector is to be used in multiagent research where direct communication is not allowed but the relative pose of the vehicles will be used for motion planning.

## Technical Approach
Apply single shot 6D object pose detection to estimate autorally vehicle pose and twist, then use these estimates to predict future state. For now, we plan to make use of the framework proposed by Tekin et. al. in [Real-Time Seamless Single Shot 6D Object Pose Prediction](https://arxiv.org/pdf/1711.08848.pdf) to track pose, then we aim to derive these pose estimates over time to obtain velocity and acceleration estimates.


## Experimental Approaches
The data collection process will consist of driving two autorally vehicles on a dirt track, with one vehicle chasing the other. The vehicles will be traveling at a low speed around a circular track. Global state estimates, containing the position and velocity of each vehicle, are gathered continuously. We will develop an auto labeller that will use global estimates to generate highly accurate positions of the vehicles in each frame for training. We will have to generate our own dataset since there is no large dataset of multiple vehicles driving autonomously on the same track.

We will use the provided implementation of the 6D pose estimate provided by the authors. The AutoRally platform and core code will be used as provided [here](https://github.com/AutoRally/autorally). We will have to implement an auto labeller ourselves along with training a network that will effectively work at the given task.

## Experimental Setup:
Stationary tracking of another vehicle in lab. Expect velocity and acceleration estimates of 0 m/s and 0m/s^2, respectively.
Stationary tracking of another vehicle on the track. Expect velocity and acceleration estimates of 0 m/s and 0m/s^2, respectively.
Slow moving tracking of vehicle while moving straight (3 m/s)
Medium speed tracking of a vehicle while moving straight (6 m/s)
Slow moving tracking of a vehicle driving around an oval track (3 m/s)
Medium moving tracking of a vehicle driving around an oval track (6 m/s)
Success for the project is defined by our ability to accurately and reliably estimate the relative pose and twist of the leading vehicle relative to the chasing vehicle. This would be a useful step towards multi-agent racing on the autorally platform.

https://arxiv.org/pdf/1909.07707.pdf 
