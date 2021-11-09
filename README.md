# FACIAL KEYPOINT DETECTION

This notebook is broken down in the following order:
* [1. Introduction](#1)
* [2. Methodology](#2)
  * [2.1 Data Acquisition](#2.1)
  * [2.2 Data Structure Overview](#2.2)
  * [2.3 Modeling Approach](#2.3)
* [3. Data Exploration](#3)
* [4. Model Performance Selection](#4)
* [5. Model Selection and Training](#5)
  * [5.1 Benchmark Model](#5.1)
  * [5.2 Custom Architecture](#5.2)
  * [5.3 MobileNetV2](#5.3)
  * [5.4 DenseNet169](#5.4)
  * [5.5 XCeption](#5.5)
* [6. Modeling Results](#6)
* [7. Explainability](#7)
* [8. Conclusions](#8)
* [9. Future Improvements](#9)

## 1. INTRODUCTION <a name='1'></a>
Computer vision is an ever growing market, expected to reach $48.6 billion by 2022. Especially in a world with an increasing supply of visual information from security systems, phones, and entertainment footage, computer vision is an area of opportunity for a range of activities. 

The gaming industry is exceptionally large; according to Statista, the video game industry generated $155 billion in revenue in 2020. This isn’t that surprising considering that there’s approximately 2.6 billion gamers worldwide in 2020, about a third of the population. I expect that as both technology and the gaming industry continues to advance, computer vision and gaming will naturally become more and more intertwined. 

One example of that is A/R gaming, where a user’s game content is integrated with their environment in real time, creating an immersive experience. We’ve already seen the success of Pokemon Go, and I fully expect that we’ll have games with visual headsets/glasses that can interact with the real world in a full immersive experience. From something like Free Guy, where the user has their customized content superimposed onto their environment, to Her, where the user can freely interact with a projection onto their environment. In whatever scenario, computer vision plays a critical role. One part that's particularly interesting is object detection, which identifies a class of image and detects it in an image. Object detection would help in recognizing the furniture in the room for A/R gaming, among other things. What about the use of key-point detection to enhance the user gaming experience? For example, let's take a look at the below image:

![augmented reality gaming](https://blogs.geniteam.com/wp-content/uploads/2020/02/Ar-pic-3-1-1024x576.jpg "Image taken from blogs.geniteam.com")

Now, imagine if there was no need for a headset. Anyone who has played games with a headset for longer than an hour can say that headgear is often clunky and needless to say, leaves marks on the face afterwards. Sure, technology may advance enough in the future to make them lightweight, but what if the headgear could be eliminated altogether? That's the goal of this project.

This project focuses on detecting human facial attributes-- specifically, five key points of the face: two eyes, nose, and the two corners of the mouth. In the context of A/R gaming, any headgear could be be avoided entirely this way by detecting a user’s field of vision. This could not only avoid some logistical production issues (like manufacturing different sizes at decent comfort levels) but also increase the user experience.

Outside of A/R gaming, facial keypoint detection could easily be used for security reasons, or safer emergency training with the use of A/R.

## 2. METHODOLOGY <a name='2'></a>

### 2.1 Data Acquisition <a name='2.1'></a>
The analysis done is based on the CelebA dataset from Kaggle. This dataset was originally collected by researchers at MMLAB, the Chinese University of Hong Kong. It contains 202,599 face images of various celebrities; from those images, there are 10,177 unique unknown identities. The dataset also includes 40 binary attributes and coordinates of 5 facial keypoints (two eyes, one nose, and two corners of the mouth.) 

This research in particular focuses on the five facial keypoints and correctly detecting them. At the current version, it will not work with the binary attributes in the data. 

https://www.kaggle.com/jessicali9530/celeba-dataset
http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

### 2.2 Data Structure Overview <a name='2.2'></a>
There are four .csv files provided with two .7z files for all the facial images. This project will work with only the facial images and one .csv file containing the coordinates for the five facial attributes.

The image folder that’s being used is the aligned and cropped image folder (“img_align_celeba.zip”), containing .JPG facial images that are already cropped to be focused on the face. These facial images are first roughly aligned using a similarity transformation according to the two eye locations, then resized to 218 by 178. 

The second file (“list_landmarks_align_celeba.csv”) contains the coordinates for the five facial keypoints for the aligned and cropped images.Originally presented by the MMLAB team as a .txt file, it’s been converted to a .csv file in the Kaggle dataset. It contains 10 integers: x- and y- coordinates for the left eye (“lefteye_x”, “lefteye_y”), right eye (“righteye_x”, “righteye_y”), nose (“nose_x”, “nose_y”), left mouth (“leftmouth_x”, “leftmouth_y”), and right mouth (“rightmouth_x”, “rightmouth_y”).

### 2.3 Modeling Approach <a name='2.3'></a>
Since this is a supervised learning problem, data cleaning will be minimal so that the model can learn from the raw data. Missing values will be dealt with, and I don’t expect to be doing any one-hot encoding for this dataset.

The data will then be fed through convolutional neural network (“CNN”) models, evaluating their performance against each other. Due to its feed-forward architecture, CNNs can analyze spatial images efficiently for image classification and detection. Rather than use it for image classification, I will use it as a regression model to predict the 5 (x, y) coordinate pairs of the facial keypoints. I plan to use my own CNN architecture, as well as use transfer learning.

After training and evaluating the models, I plan to dive deeper into their explainability with GradCAM. 

Because this is an unsupervised problem, I will be using Kaggle notebooks for their GPU use. The limitation is that Kaggle notebook memories are capped at 16 RAM. I suspect this might be an issue depending on the number of images I will use.

## 3. DATA EXPLORATION <a name='3'></a>
The distribution of the coordinates was visualized to get an idea of what they look like. It will also be good to note if the model predictions fall outside these ranges. 

A general heatmap of the location of the coordinates was also plotted for a better overall view.

The coordinates’ correlation map was also plotted. It was surprising to see that some coordinates weren’t as correlated as I thought they would be.


## 4. PERFORMANCE METRIC SELECTION <a name='4'></a>
Since this is a regression problem, MSE will be used as the model metric to serve as scoring in the training section. The main reason for MSE is the distribution of the coordinates; they are relatively normal and I don’t expect any outliers that would skew the metric. 

MSE =1ni=1n(yi-yi)2

The MSE will be evaluated for each of the ten coordinates in the test data as well as obtaining an average of all of them. 

For model comparison, a different metric will be used since the image sizes and keypoints are on different scales and MSE is scale-dependent. As such, a percentage error for each coordinate will be calculated, dividing the RMSE’s by the average and multiplying by 100.

## 5. MODEL SELECTION AND TRAINING <a name='5'></a>
Training will be done with four CNN models: one with a custom architecture formed through looking at Kaggle discussion forums, and three pre-trained models used with transfer learning.

### 5.1 Benchmark model <a name='5.1'></a>
A benchmark model will be created that consistently predicts only the average of each coordinate consistently. The evaluation metrics for this model will be used in comparison to the other models’. For a direct comparison, the image size and amount used for this benchmark model will be the same as the custom architecture model.

### 5.2 Custom architecture <a name='5.2'></a>
I will be obtaining one from looking at the Kaggle forums. In order to use a good sized image set (35,000), the images will be resized down to 65 x 80 and the keypoint coordinately respectively. 

The architecture here will consist of a few convolutional blocks, followed by dense layers.

### 5.3 MobileNetV2 <a name='5.3'></a>
MobileNetV2 is also another lightweight model (14 MB) since it’s optimized to perform on mobile devices. One of the reasons for its performance efficiency is its inverted residual structure where the residuals connections are between the bottleneck layers. Its architecture consists of the convolution layer with 32 filters, followed by 19 residual bottleneck layers.

For the transfer learning portion, imagenet weights will be used with a simple architecture added at the end of the MobileNetV2 to result in the final 10 nodes. Because MobileNetV2 was originally trained with images sized 224 x 224, the keypoints and images will also be rescaled to that size. Due to limited memory in Kaggle notebooks, the amount of images used will also be decreased.

### 5.4 DenseNet169 <a name='5.4'></a>
DenseNet is another lightweight architecture model (57 MB). Its dense architecture stems from shorter connections between layers close to the input and those close to the output. Each layer obtains additional inputs from all preceding layers, passing its own feature-maps to all subsequent layers. This alleviates the vanishing-gradient problem, strengthens feature propagation, and reduces the number of parameters.

As with MobileNetV2, I will use the imagenet weights with the same simple architecture added at the end to result in the final 10 nodes. DenseNet169 was originally trained with images sized 224 x 224, so I will be rescaling my keypoints and images to that size. Due to limited memory, I will also need to decrease the total amount of images used to 2,000 and the batch size during training to 50. To attempt to offset this, I will be increasing the epochs during training from 300 to 400.

### 5.5 Xception <a name='5.5'></a>
The model Xception is a relatively heavier model (88 MB) than the last two, but still lightweight while retaining one of top accuracies despite its lower number of parameters, compared to VGG-16. This is due to its modified depthwise separable convolution layers, where the pointwise convolution (1x1 convolution) happens before the depthwise convolution (channel-wise spatial convolution) and lacks any intermediate activation. The biggest limitation was its original image size of 299x299, using up a lot of memory. The keypoint coordinates were rescaled accordingly.

As with the other models, I will use the imagenet weights with the same simple architecture added at the end to result in the final 10 nodes. Due to limited memory, I will also need to decrease the total amount of images used to 1,000 and the batch size during training to 50. To attempt to offset this, I will be increasing the epochs during training from 300 to 400.

## 6. MODELING RESULTS <a name='6'></a>

| Model | Average MSE | 
| ---- | ---- |
|Benchmark | 1.694218 |
|Custom | 0.994353 |
|Xception | 1055.3187 |
|MobileNetV2 | 592.19005 |
|DenseNet169 | 1316.3563 |

The custom model performed the best, out of all the models. I was surprised to see the pretrained models to do so horribly, but that might be because the smaller amount of images used. At the least, I’m relieved to see that the custom model still performed better than the benchmark.

## 7. EXPLAINABILITY <a name='7'></a>
With GradCAM, I was able to take a closer look at the models’ explainability. It is important to note that explainability does not necessarily mean interpretability. Meaning cannot be exactly extracted from looking at the model’s partial actions.

For all models, I looked at three convolutional layers: the very first one, the very last one, and a middle one. I expect the models to focus on more bigger-picture detection, like edges and general shapes, in the first layer. At the same time, I expect the opposite to be true for the last layer, where the model might look at specific areas related to the coordinate.

I visualized these layers for two labels: the left eye y-coordinate and the nose x-coordinate. From the initial data visualization, it looked like they weren’t really correlated so I expected GradCAM to show that the layers would be looking at different things during the last convolution layer for these labels. 
However, for all the models, that was not the case; all three layers visualized were relatively the same for both labels. This could suggest that the models were underfitting and need more tuning. Or it may simply be how the models work, given their lack of interpretability. 

## 8. CONCLUSIONS <a name='8'></a>
The custom model was the only model that performed better than the benchmark, which was surprising. The translation of objectives of the pre-trained models could have an impact; these models were intended for image classification as opposed to a coordinate regression, though I’m unsure how much of an issue this would be. It may also have to do with the general architecture that was added at the end of these models, and a different architecture was needed. More tuning would be required to know, but with the limited resources and the high MSE, it may not seem like a worthwhile pursuit.

It was also interesting to see that with GradCAM the model is generally focusing on the same areas for different coordinates. It may be because the coordinates chosen have a higher variance than the others, but it may be GradCAM or the model potentially underfitting. 

In the end, although the custom architecture model was better than the benchmark, there could still be more work done to improve the performance. 

## 9. FUTURE IMPROVEMENTS <a name='9'></a>
The current model could perform better and be expanded into a full, executable project. Currently, the best performing model rescales the images to be pretty small, at 65x80. Perhaps the model could learn better with higher resolution images. However, that would require more resources.

Since it is meant to be used in the real world, an API or web app could be developed so that people can submit facial pictures and the model would output the picture with its facial keypoints predictions. It would need to first preprocess the image to the correct size to feed it into the model.

With additional resources, the transfer learning models could be explored further. The relatively fewer images could be a reason why they were performing so horribly compared to the custom architecture. So the total amount of images could be increased, as well as batch size or epochs during training. 

The model itself could also be expanded in breadth and depth. It could vertically improve by using the uncropped images so that the model learns to detect the general area of the area as well as pinpoint the facial keypoints. On the other hand, it could be horizontally upgraded by using the facial attributes dataset, which contains 40 binary classifications of facial characteristics like gender, beard or no beard, etc. As such, not only would the model detect the facial keypoints, but also recognize what characteristics the facial image contains.

This model could be the beginnings of many different projects. Going back to the objective of this project, it could be implemented as the initial stages in a facial keypoints detection project using video footage as opposed to only images. With computer vision, the possibilities are endless.
