# **Behavioral Cloning** 


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./tfboard.png "TensorBoard"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

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


