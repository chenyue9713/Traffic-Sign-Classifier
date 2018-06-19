# **Traffic-Sign-Classifier**



**Traffic-Sign-Classifier**

In the project, I used deep neural networks, convolutional neural networoks and image processing to classify German traffic signs. For successfully achieving the goal, several technologies were uesed:

    1. Scaling and cropping image
    2. Normalizing image data 
    3. Gray scaling
    4. Tensorflow
    5. Convolution Neural Networks
    6. LeNet 5

The LeNet-5 Convolution Neural Networks was first introduced by Yann LeCun et al. in their 1998 paper. The architecture is shown in following figure:

![png](images/loading6pic.png)

I will implement the architecture for training Traffic Sign Classifier model and modify hyperparameters to increase accuracy for validation set.

## Dataset Exploration

Before configuring and training LeNet-5 Neural Network, it is essential to have an insight about the datasets used for our Network.
   
### Load The Data

```python
import pickle


``` 

### Dataset Summary & Exploration
The pickled data is a dictionary with 4 key/value pairs:
* 'feature' is a 4D array containing raw pixel data of the traffic sign images
* 'labels' is a 1D array containing the label of the traffic sign
* 'sizes' is a list representing the orignal width and height the image
* 'coords' is a list representing coordinates of a bounding box around the sign in the image

#### Summary of the data Set

```python
n_train =

n_validation =


``` 

#### Exploratory visualization of the dataset

Matplotlib is used for ploting the some Traffic signs images and ploting the count of each sign.
```python
import matplotlib.pyplot as plt

%matplotlib inline


``` 

Also I am quite interested in percetage for each sign in dataset. 
```python
import matplotlib.pyplot as plt

%matplotlib inline


```
It looks like the the distribution for each sign in various dataset(train, validation, test) is similiar. It is important because if the distributions are quite different, the model trained by train dataset can not guarantee it will have similiar performance/accuracy.


## Design and Test a Model Architecture

#### Preprocess the Data set












    
