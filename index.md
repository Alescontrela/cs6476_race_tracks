Team Members: Alejandro Escontrela, Lishi Sun, Jason Gibson
Canonical link: https://alescontrela.github.io/cs6476_race_tracks/

# Problem Statement
To track simulated [autorally](https://arxiv.org/pdf/1806.00678.pdf) vehicle pose (position & orientation) in 3D of a leading vehicle from the on-board camera of a chasing vehicle. The system will be fed a sequence of images of a simulated environment from a monocular camera, and the output will be a real-time estimate of the leading vehicle's pose relative to the chasing vehicle. The estimated poses will be used in a multiagent scenario where inter-vehicle communication is not allowed but knowledge of the relative motion is essential. This can then be applied to real-life conditions as seen in the video below.


<p align="center">
  <a href="https://www.youtube.com/watch?v=IEXOmSZwCcA">
    <img src="https://img.youtube.com/vi/FbcGs-XoiUw/0.jpg"/>
  </a> <br/>
  Video demonstrating multiple autorally vehicles operating around a track at various velocities. 
</p>

---
# Teaser Image

<p align="center">
  <img width="60%" src="https://i.imgur.com/FR8FxmJ.png"/>
<img width="60%" src="https://i.imgur.com/udhqdjH.png"/>
  </a> <br/>
  Figure 1: Pose Estimation for a Simulated AutoRally Racer
</p>

---
# Approach

## Technical Approach

The first step in our approach is to apply the pose estimation algorithm desribed in [Real-Time Seamless Single Shot 6D Object Pose Prediction](https://arxiv.org/pdf/1711.08848.pdf). This algorithm extends the [YOLO](https://arxiv.org/pdf/1506.02640.pdf) (You Only Look Once) state-of-the-art object detection architecture to predict pose in the camera frame.

The aforementioned framework is limited to detecting object poses in a single frame. As such, useful application of this framework to autorally requires significant enhancement. First, keeping track of multiple vehicles through time is necessary when estimating the vehicle's reward functions. Additionally, obtaining a robust estimate of the lead vehicles' velocity and acceleration is required to predict their future motion.

Visual tracking of multiple vehicles can be achieved via a range of technqiues, including but not limited to [VOT](http://www.votchallenge.net/), [TLD](https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/), [GOTURN](http://davheld.github.io/GOTURN/GOTURN.html), etc. Our aim is to explore the space of object tracking and apply the technique which best fits our needs.

The first contribution of this project is to validate the ability to track a vehicle in a simulated environment and then test on a real-life video stream. The second is the pose predictor that takes the aforementioned tracking information from each image and predicts pose. Due to time contraints, we were only able to generate successful results for the simulated dataset. The performance on the real-world dataset had marginal success, but bounced around due to false detections.

<p align="center">
  <img width="60%" src="ouput_viz.png"/> <br/>
  Figure 2: Sample outputs from the 6D pose estimation algorithm.
</p>

### API Specification

**Inputs:**
  * A real-time stream of simulated camera images obtained from a monocular camera onboard a chasing autorally vehicle containing a single lead vehicle.

**Outputs:**
  * An output for the detected leading vehicle, containing:
    * The vehicle's XYZ position relative to the camera frame.
    * The vehicle's predicted location within the 2D image
    * The vehicle's predicted 3D orientation

## Data Gathering
The simulation of the autorally racer was generated and rendered as a video that will be fed into the AutoLabeler. Global state estimates, containing the position and velocity of each vehicle, are gathered continuously via a Factor Graph fix lag smoother (described in detail later). The auto labeller that uses these global estimates to generate highly accurate positions of the vehicles in each frame for training.

The simulation dataset was collected by running Best Response MPPI with a following distance of 1 meter. This keeps the two vehicles within about a meter of each other and since their target speed was identical no passing maneuvers were attempted. Therefore the chased vehicle will always remain in the field of view of the chasing vehicle. Signifigant improvements had to be made to the AutoRally simulator to enable efficient usage of the multi-vehicle aspect of the simulator. Gazebo allows usage of multiple vehicles out of the box but getting the proper configuration to work with the AutoRally code base required modificaition to the underlying controller code to ensure all ROS messages were recieved and published correctly. 

The intital round of data was collected during the implementation of the paper, [Autonomous Racing with AutoRally Vehicles and Differential Game](https://arxiv.org/pdf/1707.04540.pdf) by Williams et all, provided the initial data and inspiration for this project. 

We will use the Single Shot 6D Object Pose Prediction implementation provided by the paper's authors, the AutoRally platform and core code as provided [here](https://github.com/AutoRally/autorally), and implement an auto-labeler that will provide the data required to train the 6D pose estimation network. We will be working off of a fork of the AutoRally repo [here](https://github.com/JasonGibson274/AutoRally) in the autorally_vision package.

We further test its generalizability by applying the AutoLabeller to a real-world video stream of an AutoRally races. The accuracy of the state estimate is a signifigant issue when using the AutoLabeller, we often get images like we show below where the label is off since the orientation estimate is incorrect for the chasing vehicle.

<p align="center">
  <img width="60%" src="https://i.imgur.com/Y7zqGEw.jpg"/> <br/>
  Figure 3: Incorrect State estimate resulting in incorrect labels.
</p>![]()

---

# Experiments and results
## Experiment Goals

Success for the project is defined by our ability to accurately and reliably track the leading vehicle in real time as well as accurately and reliably estimating its pose. This would be a useful step towards multi-agent racing on the autorally platform. Uncertainty in our outcomes would be measured by our system's error on a test set generated by the auto-labeler. This error metric will incorporate tracking and pose estimate errors.

## Data Collection
### Overview
We generated the simulated datasets by running two version of MPPI in the gazebo simulator for AutoRally. This simultaneously creates both the video stream and pose, which we then package into bag files. These are then hosted on Georgia Tech's Box due to their size.

### Data Features

Each dataset is composed of a a bag file that contains all simulated sensor data, video images, and pose estimates of the Autorally vehicle going around a track. The Factor Graph data containing the vehicle's global state estimates is synchronized with the video which is then fed into the AutoLabeller to generate the bounding box to track the vehicle.

## AutoLabeller
### Approach
The AutoLabeler was built from scratch for the AutoRally project. As for contributions for CS 6476 in particular,  we were able to test this software on both simulated and real world data to confirm its validity. 

Adjustments in the bouding box size between the simulation set and the real world set was extensively tested until consistent matches were made between the two sets using the same parameters. Using the output from this step, we can then train a predictor to estimate pose for a simulated vehicle. Extension of the predictor to real-data sets will be done at a later time.

The quality of the results are manually verified by members of the team to ensure reliable automatic labeling for the real world set. The difference in corner pixel locations was used for the simulated sets, since those are better quantified and known. The best results from this section are then used as the training set later on.

### Results on Simulated Dataset
The following image shows the results of the AutoLabller applied to the simulated vehicle. Multiple iterations showed close matching of the bounding box with the vehicle across different velocities and acclerations, which gave enough confidence to proceed with testing with the real vehicle below. 
<p align="center">
  <img width="60%" src="https://i.imgur.com/pY09QsV.jpg"/> <br/>
  Figure 3: Bounding Box for Simulated Vehicle
</p>

### Results on Real-Life Dataset
The following vidoes show the results of the AutoLabeller applied to two bag files on different days in sunny weather for a real AtuoRally racer.

<p align="center">
  <a href="https://www.youtube.com/watch?v=Fc9z1R0G2gA">
    <img src="https://img.youtube.com/vi/Fc9z1R0G2gA/0.jpg"/>
  </a> <br/>
  Video demonstrating autolabeler outputs overlaid on two AutoRally vehicles moving at high velocities.
</p>

Though, as the video also shows, there is some work that remains to be done in cleaning up the bounding box data generated to produce consistent results in our predictor downstream.

<p align="center">
  <img width="60%" src="https://i.imgur.com/mcQnnE4.png"/> <br/>
  Figure 4: Bounding Box for Vehicle in Above Video.
</p>

<p align="center">
  <img width="60%" src="https://i.imgur.com/wTAsujr.png"/> <br/>
  Figure 5: Bounding Box for Vehicle in Above Video.
</p>
 
## Pose Estimation Neural Network

The bounding box generated above paired with the images from the dataset will then be used to predict for the pose of the autorally vehicle(s). The predictor used is a deep convolutional neural network that is based largely on  the [Real-Time Seamless Single Shot 6D Object Pose Prediction](https://arxiv.org/pdf/1711.08848.pdf) to predict bounding boxes around the AutoRally vehicle as described in the [Technical Approach](#Technical-Approach).

The base approach extracts the 2D pose from a 3D bounding box. To ensure that the training process for our custom dataset goes smoothly, we set up a validation pipeline to analyze model outputs and performance.  We set up a [small Colab notebook](https://colab.research.google.com/drive/1BabqYyXQSv527-GPTiWV3NxxZRzLcbjN) to pass test images into the singleshotpose network and visualize the output. Here is an example output:

<p align="center">
  <img width="60%" src="https://i.imgur.com/GsSshBO.png"/> <br/>
  Figure 6: Sample network prediction for a model trained on the LINEMOD dataset.
</p>
 
We noticed that this network occassionally predicts inaccurate orientations, as is evident here:
<p align="center">
  <img width="60%" src="https://i.imgur.com/DgfNjaM.png"/> <br/>
  Figure 7: Sample network prediction for a model trained on the LINEMOD dataset with some orientation error.
</p>

View more sample outputs [here](https://colab.research.google.com/drive/1BabqYyXQSv527-GPTiWV3NxxZRzLcbjN).

Once we were able to achieve a close match between this smaller validation set on stationary objects, we extended this to the our training dataset. Whereas the original package only predicted for stationary images, we extended it to be able to track the lead vehicle and predict its poses in real time.

Additionally, using data augmentation, we are able to generate a far larger training set from the set of image frames we have collected. Each frame from the training set is rotated slightly and has noise added, with the corresponding true coordinates adjusted for the change. This trains our predictor to be robust to noise and rotation.

### Error Metric
To measure performance, we established the ground truth of the pose by the pre-defined values in our simulated racer and the measured values on the real vehicle and compared them with the values generated by our predictor.

#### Pose Error
To assign empircal measures to the closeness of the orientation to the ground truth, we used the following measures.
* *Mean Corner Error* (px)
* *Accuracy in 2D Projection* (% overlap)
* *Accuracy in 3D Projection* (% overlap)

### Parameters Tuning

#### Parameters Tested

We used weights for the DarkNet architecture that were pre-trained on ImageNet to avoid expensive/excessive retraining and tuning. Running it in this manner initially produced accurate results, so further adjustment of weighting/tuning for this network was deemed as a future extension of this project.

## Quantitative Performance

The single-shot pose CNN was able to train the model quite readily on the data gathered from simulation. The following plot shows the training loss for each batch of 20 images. Training loss is measured as the mean squared error of each of the bounding box coordinates to the true coordinates.
![](https://i.imgur.com/oQdfCwD.png)

The results are quite promising, as the training loss converges after 1000 batches to a relatively low level of error.

The error for the real world dataset is higher on average. This reflects the fact that it is harder to train the model on actual data. The first reason that is more difficult to generalize to real world images is that errors in the state estimate for the vehicle pose propagate to the autolabeller, which means that labels are not always accurate. Additionally, dirt, loss of camera focus, blur due to vibration, and other factors cause significant variance in the images fed into the singleshotpose model. This leads to greater difficulty in training. The following plot shows the training loss for the real world dataset as a function of the batch number. Notice that a) the loss is higher than in the simulated dataset and b) the training loss has a higher variance for the real world dataset than for the simulated dataset.

![](https://i.imgur.com/n7IhOW2.png)


## Qualitative Performance

The following video shows both the auto-labeler followed by the pose predictor:

<p align="center">
  <a href="https://youtu.be/oWBPvHWlFuQ">
    <img src="https://img.youtube.com/vi/oWBPvHWlFuQ/0.jpg"/>
  </a> <br/>
  Video demonstrating autolabeler and singleshotpose model predictions.
</p>

![](https://youtu.be/oWBPvHWlFuQ)


As we can see, the autolabeller was able to capture the bounding box of the vehicle reliably in the simulated environment. However, we can see some translation error in the real world run of the labeler. This may be due to a much noisier environment surrounding the target vehicle. 

The run of the predictor on the simulated car shows very tight grouping between the true and predicted poses. This validates the high accuracy we observed in the quantitative section. There was more error in the run on the real-life dataset due to translation errors in the original bounding boxes upstream. Of note is that there was a larger degree of error in the x-dimension because we assumed the height of the car relative to the camera to be constant for the predictor.

# Conclusion
Over the course of this project, our goal was to  create a predictor for an simulated Autorally vehicle's  pose. The novel approach we have proposed adapts an AutoLabeller and CNN in conjuction to predict these values. We have shown that our approach reliably predicts for the pose of a simulated vehicle in real-time. Additionally, we have shown that the auto-labeler can be effectively used to track real-world vehicles. 

Previous updates noted that twist was also to be predicted for in this manner, but due to time constraints, only pose was reliably captured in the allotted time.

# Future Work
To expand on this in future works, the next step would be to apply the predictor to a real-world video of an AutoRally vehicle. Doing so would involve more extensive collection of bounding boxes from real datasets. Controlling for occlusions and noise will be a future challenge as well within this problem space. Further extensions include adding predictions for linear and angular velocity.

Additionally, extensions to this project can focus on improving AutoLabeler accuracy by more effectively ignoring false positives as they occur. Additionally multiple cameras may be used on a chase vehicle to get depth information to generate a richer training data set.

The final extension would be to incorporate a factor graph to the actual vehicle tracking task. We would have the vehicle poses and velocities as variables in the factor graph. We would use the GTSAM factor graph library to estimate the locations. The state estimator would be a unary factor on the chasing vehicle pose and velocity as well as connecting successive states (motion model). The camera detections would be a between factor on the chasing vehicle pose and the detected vehicle pose. Finally the motion of the chasing vehicle would be constrained by the dynamics model applied to the other vehicle. We can do this because in a game theoretic setting we would have an estimate of the controk vector of the other vehicle.

## Source Code

Singleshotpose Pipeline:
* Source Code: https://drive.google.com/drive/folders/1L010K_HJssbSyrO2Z6kS01TPyI75gr91?usp=sharing
* Training notebook: https://colab.research.google.com/drive/1-5-YQNY90pWLXitYQs5HsIHCf5urrYin
* Validation notebook: https://colab.research.google.com/drive/1r95hQgiW9iGAhHYawdIGP5LluTulsZJw

Auto-Labeler Code: https://github.com/JasonGibson274/autorally/tree/vehicle_tracker/autorally_vision

## References
Tekin, Bugra, Sudipta N. Sinha, and Pascal Fua. "Real-time seamless single shot 6d object pose prediction." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.

Redmon, Joseph, et al. "You only look once: Unified, real-time object detection." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

Williams, Grady, et al. "Autonomous racing with autorally vehicles and differential games." arXiv preprint arXiv:1707.04540 (2017).
