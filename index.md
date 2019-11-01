Team Members: Alejandro Escontrela, Lishi Sun, Jason Gibson
Canonical link: https://alescontrela.github.io/cs6476_race_tracks/

# Problem Statement
To track [autorally](https://arxiv.org/pdf/1806.00678.pdf) vehicle pose (position & orientation) and twist (linear and angular velocity) in 3D of a leading vehicle from the on-board camera of a chasing vehicle. The system will be fed a sequence of images from a monocular camera, and the output will be a real-time estimate of the leading vehicle's pose and twist relative to the chasing vehicle. The estimated poses will be used in a multiagent scenario where inter-vehicle communication is not allowed but knowledge of the relative motion is essential. 


<p align="center">
  <a href="https://www.youtube.com/watch?v=IEXOmSZwCcA">
    <img src="https://img.youtube.com/vi/FbcGs-XoiUw/0.jpg"/>
  </a> <br/>
  Video demonstrating multiple autorally vehicles operating around a track at various velocities. 
</p>

---

# Approach

## Technical Approach

The first step in our approach is to apply the pose estimation algorithm desribed in [Real-Time Seamless Single Shot 6D Object Pose Prediction](https://arxiv.org/pdf/1711.08848.pdf). This algorithm extends the [YOLO](https://arxiv.org/pdf/1506.02640.pdf) (You Only Look Once) state-of-the-art object detection architecture to predict pose in the camera frame.

The aforementioned framework is limited to detecting object poses in a single frame. As such, useful application of this framework to autorally requires significant enhancement. First, keeping track of multiple vehicles through time is necessary when estimating the vehicle's reward functions. Additionally, obtaining a robust estimate of the lead vehicles' velocity and acceleration is required to predict their future motion.

Visual tracking of multiple vehicles can be achieved via a range of technqiues, including but not limited to [VOT](http://www.votchallenge.net/), [TLD](https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/), [GOTURN](http://davheld.github.io/GOTURN/GOTURN.html), etc. Our aim is to explore the space of object tracking and apply the technique which best fits our needs.

Accurately estimating the linear/angular velocity and linear acceleration of the lead vehicles will require the application of filtering techniques, to ensure that the estimates are robust to noise. 

<p align="center">
  <img width="60%" src="ouput_viz.png"/> <br/>
  Figure 1: Sample outputs from the 6D pose estimation algorithm.
</p>

At a high level, the system's API would behave as follows:

**Inputs:**
  * A real-time stream of camera images obtained from a monocular camera onboard a chasing autorally vehicle containing a variable number of leading vehicles.

**Outputs:**
  * An output for each detected leading vehicle, containing:
    * A unique identifier for the vehicle.
    * The vehicle's XYZ position relative to the camera frame.
    * The vehicle's linear velocity and acceleration relative to the camera frame.
    * The vehicle's angular velocity relative to the camera frame.


## Data Gathering
The data collection process will consist of driving two autorally vehicles on a dirt track, with one vehicle chasing the other. The vehicles will be traveling at a low speed around an oval track. Global state estimates, containing the position and velocity of each vehicle, are gathered continuously via a Factor Graph fix lag smoother (described in detail later). We will develop an auto labeller that will use these global estimates to generate highly accurate positions of the vehicles in each frame for training.

The intital round of data was collected during the implementation of the paper, [Autonomous Racing with AutoRally Vehicles and Differential Game](https://arxiv.org/pdf/1707.04540.pdf) by Williams et all, provided the initial data and inspiration for this project. Here two vehicles were autonomously navigating the track

We will use the Single Shot 6D Object Pose Prediction implementation provided by the paper's authors, the AutoRally platform and core code as provided [here](https://github.com/AutoRally/autorally), and implement an auto-labeler that will provide the data required to train the 6D pose estimation network. We will be working off of a fork of the AutoRally repo [here](https://github.com/JasonGibson274/AutoRally).

The auto-labeler will produce leading vehicle's bounding box vertices in the camera frame given the global state estimates for multiple vehicles.

We will have to generate our own dataset since there is no large dataset of multiple vehicles driving autonomously on the same track. Our data gathering process will be carried as follows:
1. _Stationary_ tracking of another vehicle on the track.
    * Expect velocity and acceleration estimates of 0 m/s and 0m/s^2, respectively.
   
2. Tracking of a vehicle in a straight line at low speeds with constant velocity.
    * Expect velocity and acceleration estimates of 3 m/s and 0m/s^2, respectively.

3. Tracking of a vehicle in a straight line at medium speeds with constant velocity.
    * Expect velocity and acceleration estimates of 6 m/s and 0m/s^2, respectively.
  
4. Tracking of a vehicle in a straight line with constant acceleration.
    * Expect velocity to increase from 0 to 6m/s and acceleration to remain constant at 1m/s^2.
 
5. Tracking of a vehicle in an oval track line at low speeds with constant velocity.
    * Expect velocity and acceleration estimates of 3 m/s and 0m/s^2, respectively.
 
6. Tracking of a vehicle in an oval track line at medium speeds with constant velocity.
    * Expect velocity and acceleration estimates of 6 m/s and 0m/s^2, respectively.

7. Tracking of a vehicle in an oval track with constant acceleration.
    * Expect velocity to increase from 0 to 6m/s and acceleration to remain constant at 1m/s^2.


# Experimental Outcomes
Success for the project is defined by our ability to accurately and reliably estimate the relative pose and twist of the leading vehicle relative to the chasing vehicle as well as track the leading vehicle over time. This would be a useful step towards multi-agent racing on the autorally platform. Uncertainty in our outcomes would be measured by our system's error on a test set generated by the auto-labeler. This error metric will incorporate tracking, pose, and twist estimate errors.

---

# Progress Update: Experiments and results
## Data Collection
### Overview
We were able to successfully gather datasets for the aforementioned cases 1-7. Datasets are hosted at Georgia Tech Box due to their size.


### Data Features

Each dataset is composed of a a bag file that contains all sensors, camera images, and pose estimates of the Autorally vehicle going around a track. The Factor Graph data containing the vehicle's global state estimates is synchronized with the video which is then fed into the AutoLabeller to generate the bounding box to track the vehicle.

## AutoLabeller
### Results
Using the data from the previous section, we were able to use the autorally code to generate an 8-point bounding box and centroid for a lead vehicle. The following vidoes show the results of the AutoLabeller applied to two bag files on different days in sunny weather.

<p align="center">
  <a href="https://www.youtube.com/watch?v=Fc9z1R0G2gA">
    <img src="https://img.youtube.com/vi/Fc9z1R0G2gA/0.jpg"/>
  </a> <br/>
  Video demonstrating autolabeler outputs overlaid on two AutoRally vehicles moving at high velocities.
</p>


This allows us to give each vehicle a unique tag that the bounding box will be associated with. Given that we are able to calculate these points over time, we can give them as inputs to a CNN to then predict the vehicle's positon, linear velocity, and angular velocity relative to the camera frame. Though, as the video also shows, there is some work that remains to be done in cleaning up the bounding box data generated to produce consistent results in our predictor downstream.

#### Example of successful bounding box
<p align="center">
  <img width="60%" src="https://i.imgur.com/mcQnnE4.png"/> <br/>
  Figure 2: Bounding Box for Vehicle in Above Video.
</p>

<p align="center">
  <img width="60%" src="https://i.imgur.com/wTAsujr.png"/> <br/>
  Figure 3: Bounding Box for Vehicle in Above Video.
</p>



### Discussion of Error

There are instances in which the auto-labeller produces labeling that does not contain the vehicle within the bounding box. The current cause is error within Z-dimensional estimation. We plan to address this by fixing the height of the vehicles and not using the estimates generated. The AutoRally estimator works well in the x,y plane but has never been tested of meant to correctly estimate z. Since the platforms are mostly on the ground at the same approximate relative height this should not be a large problem.

The second error is when the state estimator diverges due to crashes or other various issues. Running hardware is difficult, in fact running two cars is more than twice as hard. The data we have been working with for training is during experimental runs of the new algorithm BR-MPPI and such there is frequent crashes. When that happens the factor graph tends to diverge and result in an invalid state. Building a crash tolerant estimator would require more data and time than is within the scope of this project. For now, we shall treat this as error inherent in the problem and will manually clean data as needed.

The third issue is slight inaccuracies in state estimate and camera calibration parameters. The factor graph estimator does not have any global measurements of heading and so it tends to drift over time if the vehicle is not moving fast enough, like in these experiemnts. The other values vary in accuracy through out the experiments we have data for. 

#### Examples of Bounding Box Errors

<p align="center">
  <img width="60%" src="https://i.imgur.com/NcszukC.png"/> <br/>
  Figure 4: Bounding box where no vehicle is present.
</p>

<p align="center">
  <img width="60%" src="https://i.imgur.com/41aRMx0.jpg"/> <br/>
  Figure 5: Error in bounding box post-crash.
</p>

  
## Pose Estimation Neural Network

To ensure that the training process for our custom dataset goes smoothly, we set up a validation pipeline to analyze model outputs and performance. As described in [Technical Approach](#Technical-Approach), we make use of the [Real-Time Seamless Single Shot 6D Object Pose Prediction](https://arxiv.org/pdf/1711.08848.pdf) to predict bounding boxes around the AutoRally vehicle. We set up a [small Colab notebook](https://colab.research.google.com/drive/1BabqYyXQSv527-GPTiWV3NxxZRzLcbjN) to pass test images into the singleshotpose network and visualize the output. Here is an example output:

<p align="center">
  <img width="60%" src="https://i.imgur.com/GsSshBO.png"/> <br/>
  Figure 6: Sample network prediction for a model trained on the LINEMOD dataset.
</p>
 
We noticed that this network sometimes fails to predict accurate orientations, as is evident here:
<p align="center">
  <img width="60%" src="https://i.imgur.com/DgfNjaM.png"/> <br/>
  Figure 7: Sample network prediction for a model trained on the LINEMOD dataset with some orientation error.
</p>

View more sample outputs [here](https://colab.research.google.com/drive/1BabqYyXQSv527-GPTiWV3NxxZRzLcbjN)
 
