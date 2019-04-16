# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains starting files for the Behavioral Cloning Project.

In this project, you will use what you've learned about deep neural networks and convolutional neural networks to clone driving behavior. You will train, validate and test a model using Keras. The model will output a steering angle to an autonomous vehicle.

We have provided a simulator where you can steer a car around a track for data collection. You'll use image data and steering angles to train a neural network and then use this model to drive the car autonomously around the track.

We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Behavioral-Cloning-P3/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

To meet specifications, the project will require submitting five files: 
* model.py (script used to create and train the model)
* drive.py (script to drive the car - feel free to modify this file)
* model.h5 (a trained Keras model)
* a report writeup file (either markdown or pdf)
* video.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)

This README file describes how to output the video in the "Details About Files In This Directory" section.

Creating a Great Writeup
---
A great writeup should include the [rubric points](https://review.udacity.com/#!/rubrics/432/view) as well as your description of how you addressed each point.  You should include a detailed description of the code used (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

The following resources can be found in this github repository:
* drive.py
* video.py
* writeup_template.md

The simulator can be downloaded from the classroom. In the classroom, we have also provided sample data that you can optionally use to help train your model.

## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.

#### Why create a video

1. It's been noted the simulator might perform differently based on the hardware. So if your model drives succesfully on your machine it might not on another machine (your reviewer). Saving a video is a solid backup in case this happens.
2. You could slightly alter the code in `drive.py` and/or `video.py` to create a video of what your model sees after the image is processed (may be helpful for debugging).

### Tips
- Please keep in mind that training images are loaded in BGR colorspace using cv2 while drive.py load images in RGB to predict the steering angles.


## Project Notes

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is a regression model to predict the steering.

I first tried a simple CNN model with two conv-2d layers and a FC layer, but it didn't really work very well. The car drove to the river or off the road pretty easily.

The final model consists of an imagenet pretrained inception-v3 model (top removed, model.py line 123-125) with a global average pooling layer and a 64-node fully connected layer.

I added a Lambda layer on top of the model to: 1) crop the image, 2) resize the image to 139x139, 3) normalize the image to [-0.5, 0.5].

#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layer at the end in order to reduce overfitting (model.py lines 128).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 140-142). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer with default learning rate 0.001, so the learning rate was not tuned manually (model.py line 134).

I tuned the batch size. The largest I can fit into my GPU memory is 128.

I aso tuned a few different FC layer size, and decided to keep the FC layer relatively small to save model size.

#### 4. Get training data

I used the simulator to collect some training data, but eventually I only used the data provided by the course, which can already train a decent model to finish a lap. However, it is not generalized enough to run the track 2 in the simulator. If I have more time, I will collect more training data.

#### 5. Training data preprocess and augmentation

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, but added the correction steering angle for the left and right cameras (both 0.2,  model.py line 77-80)

I also flipped the images and also flipped the steering direction (model.py line 88-91). The previous model didn't use flipped images, and the car was tending to drive on the right side. After I used the flipped images, this problem was solved.

#### 6. Creation of the Training Set

I wrote a function training_data_generator() to yield (image_batch, label_batch, weight_batch), and each batch contains 128 images.

In my initial model, the car didn't drive well in sharp curves, so I increased the weight for the images with large steering angle (model.py line 82). Because most of the steering is between -0.5 and 0.5, so the weight is defined as:  weight = 5 * (abs(label) + 0.2). For example, if the steering is 0, the weight is 1. If the steering is 0.5 or -0.5, then the weight will be 5.

I split the training/validation/test to 80/10/10 percent respectively.

#### 7. Training Process

I added an early_stop callback (line 147) to stop the training once there is no improvement in 5 epochs. The training finished at 65 epochs. Please note I used a small epoch, which only covers one camera, so that I can have more epochs to get smooth curves in tensorboard.

I used a checkpoint callback to save the best model based on val_loss (line 149). I also added a tensorboard callback to monitor the loss (mse) and mae.

This is my final results:
   - Training set (80%): loss: 0.0130 - mean_absolute_error: 0.0637
   - Validation set (10%): val_loss: 0.0309 - val_mean_absolute_error: 0.0949
   - Test set (10%): test_loss: 0.02116 - test_mean_absolute_error: 0.09699

The tensorboard gave me the following plots. Both loss and mae are dropping consistently on both training and validation set. The overfitting issue is not too bad in this model.

![alt text][image1]

#### 8. Results

After failure several times in the early models, finally the model can drive the cars around and around. I left it running for more than 10 minutes and it was still on track. Please check the video.mp4 for the results.

It is certainly not perfect, and cannot work well on track 2, but it is a good start. If I have more time, I will collect more data, and try different model architecture, including LSTM+CNN.


