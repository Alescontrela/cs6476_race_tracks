# Problem Statement
To track [autorally](https://arxiv.org/pdf/1806.00678.pdf) vehicle pose (position & orientation) and twist (linear and angular velocity) in 3D of a leading vehicle from the on-board camera of a chasing vehicle. The system would be fed a sequence of images from a monocular camera, and the output would be a real-time estimate of the leading vehicle's pose and twist relative to the chasing vehicle. The estimated poses will be used in a multiagent scenario where inter-vehicle communication is not allowed but knowledge of the relative motion is essential.


<p align="center">
  <a href="https://www.youtube.com/watch?v=FbcGs-XoiUw">
    <img src="https://img.youtube.com/vi/FbcGs-XoiUw/0.jpg"/>
  </a> <br/>
  Video demonstrating a single autorally vehicle operating around a track at verious velocities.
</p>

---

# Approach

## Technical Approach

The first step in our approach is to apply the pose estimation algorithm desribed in [Real-Time Seamless Single Shot 6D Object Pose Prediction](https://arxiv.org/pdf/1711.08848.pdf). This algorithm extends the [YOLO](https://arxiv.org/pdf/1506.02640.pdf) (You Only Look Once) state-of-the-art object detection architecture to predict pose in the camera frame.

The aforementioned framework is limited to detecting object poses in a single frame. As such, useful application of this framework to autorally requires significant enhancement. First, keeping track of multiple vehicles through time is necessary when estimating the vehicle's reward functions. Additionally, obtaining a robust estimate of the lead vehicles' velocity and acceleration is required to predict their future motion.

Visual tracking of multiple vehicles can be achieved via a range of technqiues, including but not limited to [VOT](http://www.votchallenge.net/), [TLD](https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/), [GOTURN](http://davheld.github.io/GOTURN/GOTURN.html), etc. Our aim is to explore the space of object tracking and apply the technique which best fits our needs.

Accurately estimating the linear/angular velocity and linear acceleration of the lead vehicles will require the application of filtering techniques, to ensure that the estimates are robust to noise. 



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
