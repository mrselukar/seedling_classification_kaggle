

# Machine Learning Engineer Nanodegree
## Capstone Project
Mayur Selukar
December 21st, 2018

## I. Definition


### Project Overview

[//]: # (Kaggle: https://www.kaggle.com/c/plant-seedlings-classification)
[//]: # (Why weeds: http://www.environment.gov.au/biodiversity/invasive/weeds/weeds/why/index.html)

Weeds are among the most serious threats to the natural environment and primary production industries. They displace native species, contribute significantly to land degradation, and reduce farm and forest productivity.

Invasive species, including weeds, animal pests, and diseases represent the biggest threat to our biodiversity after habitat loss. Weed invasions change the natural diversity and balance of ecological communities. These changes threaten the survival of many plants and animals as the weeds compete with native plants for space, nutrients, and sunlight.

Weeds typically produce large numbers of seeds, assisting their spread, and rapidly invade disturbed sites. Seeds spread into natural and disturbed environments, via wind, waterways, people, vehicles, machinery, birds and other animals.

The ability to differentiate a weed from a crop seedling effectively can mean better crop yields and better stewardship of the environment.

The Aarhus University Signal Processing group, in collaboration with the University of Southern Denmark, has released a dataset containing images of approximately 960 unique plants belonging to 12 species at several growth stages.

The dataset is hosted on [Kaggle](www.kaggle.com) and is free to download. 
Please visit this [link](https://www.kaggle.com/c/plant-seedlings-classification/data) to get the data via kaggale.

### Problem Statement

This is a multiclass classification problem with 12 classes representing  different plant species
Input is a given image and the goal is to classify its species.

I will be tackling this as an Image Classification problem and plan to use the CNN deep learning model.
Further on I will use the transfer learning technique to improve accuracy. Data augmentation will also be performed to make the model more generalized and accurate.

The target here is one of the following 12 species
- Black-grass
- Charlock
- Cleavers
- Common Chickweed
- Common wheat
- Fat Hen
- Loose Silky-bent
- Maize
- Scentless Mayweed
- Shepherds Purse
- Small-flowered Cranesbill
- Sugar beet   


### Metrics

Submissions are evaluated on MeanFScore, which at Kaggle is actually a micro-averaged F1-score.

Given positive/negative rates for each class k, the resulting score is computed this way:  

$$ Precision_{micro} =  \frac{\sum_{k \in C} TP_k}{\sum_{k \in C} TP_k + FP_k}  $$

$$ Recall_{micro} =  \frac{\sum_{k \in C} TP_k}{\sum_{k \in C} TP_k +  FN_k}  $$  

F1-score is the harmonic mean of precision and recall

$$ MeanFScore = F1_{micro}= \frac{2 Precision_{micro} Recall_{micro}}{Precision_{micro} + Recall_{micro}} $$  

[For Reference Click here](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)


## II. Analysis


### Data Exploration

There are 12 total categories.  
Total Images are 4750  
There are 3800 total training images.  
There are 950 total testing images.  
The resolution of the images varies quite a bit with higher resolution ones being 4000x3000 px and smaller ones being 400x300 px.

The data is stored in just one directory train and no separate testing data is provided 
In order to avoid loading all of the data into memory, the data was separated into a train and a testing folder with respective subdirectories for the classes. The code present in seperator_capstone.inpyb and explained later in data preprocessing section.


*Images Per category*  
Black-grass 217 images  
Charlock 311 images  
Cleavers 225 images  
Common Chickweed 490 images  
Common wheat 174 images  
Fat Hen 379 images  
Loose Silky-bent 528 images  
Maize 170 images  
Scentless Mayweed 414 images  
Shepherds Purse 192 images  
Small-flowered Cranesbill 398 images  
Sugar beet 302 images  

The dataset is highly unbalanced to combat this a number of different strategies can be applied like undersampling the data set, image augmentation to balance the underrepresented class or we can also calculate the confusion matrix and the f1 score of the model these will give us a better understanding about the working of the model.


### Exploratory Visualization

<div style="text-align: center;margin-bottom:-22px; font-style: italic;font-size: 18px">
Sample Images
</div>

![12 sample images per cateogery](https://lh3.googleusercontent.com/qbWrB7DacbmXnLMILlM_42hk2tV-mRyL_0s1PFbtlEdfR2yp2PL3ACRhAOMTEWuaG9JabZY9vQA76RjFzTHK=w1920-h899-rw)

_Some observations:_  
All images are not having the same background and some have what appears to be barcodes in the background, we only need to detect saplings these inconsistent backgrounds will cause our model to take the background as a feature and get the wrong label for the given input.  
Due to this nature of the dataset, some performance will be lost 
This can be dealt with by masking and is discussed in future works.
I have used 80:20 as the train test and train validation split 

Plotting the no of samples per category 
![Samples per category](https://lh4.googleusercontent.com/45m0MYPLVGUb9-5j3uFVxx76MtcmeHIEH7AQzUQ_4JHGI5UynJD7zVXOpfb9heIGcTvAPylRWtJ-Ah34q2is=w1920-h899-rw)
The bar graph above represents the unbalanced nature of the dataset with some categories having close to 500 samples and some under 200.

### Algorithms and Techniques

In a regular neural network, the input is transformed through a series of hidden layers having multiple neurons. Each neuron is connected to all the neurons in the previous and the following layers. This arrangement is called a fully connected layer and the last layer is the output layer. In Computer Vision applications where the input is an image, we use convolutional neural network because the regular fully connected neural networks don’t work well. This is because if each pixel of the image is input then as we add more layers the amount of parameters increases exponentially.  
Convolutional neural networks have been some of the most influential innovations in the field of computer vision. 2012. Image classification is the task of taking an input image and outputting a class or a probability of classes that best describes the image this is best handled by the modern CNN so I will be using those
The Convolution part of the CNN is built using two main components Convolution layers and Pooling Layers 
The convolution layers increase depth by computing op of neurons connected to local layers and pooling layers perform downsampling   
| Name                          | Function                                                                                                                                                                                                                                                                     |
|-------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Input Layer (WxHxD)           | Non-Computing Layer  Represents the size of the input                                                                                                                                                                                                                        |
| Dense (Fully Connected Layer) | Dense implements the operation:  output = activation(dot(input, kernel) + bias)  where activation is the element-wise activation function passed as the activation argument, kernel is a weights matrix created by the layer, and bias is a bias vector created by the layer |
| Activation Function           | the activation function of a node defines the output of that node, or "neuron," given input or set of inputs. It can be applied as either an argument to dense layer or an activation layer                                                                              |
| Flatten                       | Flattens the Input                                                                                                                                                                                                                                                          |
| Convolution Layers            | Computer the op of the neurons connected to the local regions, By computing the dot product between the weights(filters) and the small region they are connected to.                                                                                                        |
| Pooling Layers                | further condense the spatial size of the representation to reduce the number of parameters and computation in the network                                                                                                                                                    |

The solution is built with CNN's as feature extractor and Logistic REgression as Classifier 
The Output of the convolution part of the CNN is given to Log Reg model and a 2 layer Dense net in case of the benchmark model.




### Benchmark
In this section, you will need to provide a clearly defined benchmark result or threshold for comparing across performances obtained by your solution. The reasoning behind the benchmark (in the case where it is not an established result) should be discussed. Questions to ask yourself when writing this section:
- _Has some result or value been provided that acts as a benchmark for measuring performance?_
- _Is it clear how this result or value was obtained (whether by data or by hypothesis)?_

As my Bench Mark model, I used the First approach i.e A 2 layer dense net connected to the VGG16s Convolutional part
To set the bar high. Image Augmentation was also performed and the model was run for 50 epochs
The F1 score was found to be 0.8011
The summary of the model is as follows   
   
![Model Summary](https://lh6.googleusercontent.com/-DoWtK2J8CwaS1VkDBbz0llVwyLXm398bqzG5TSxAOKZsMKxAnvWSg72wZAFCjf4oM7TudfTuY8dFONrqA9G=w1920-h948-rw)


## III. Methodology

### Data Preprocessing

__Restructuring the dataset__
- The dataset was downloaded from Kaggle and then extracted into the current working directory  
- train folder was renamed to train1 and the data was separated into train and test in the ratio of 80:20
- These separated datasets were then stored into respective test and train directory maintaining the subdirectory structure.  _This was done in order to perform image augmentation_   
After loading the dataset from train1 using load_dataset function
the datasets were split using the test_train_split and seed = 42
Thse were then stored using the code below
![](https://lh6.googleusercontent.com/z5V53uJUIoQIzKF4TmOQ36srtkcz8B2NfG4heR4JoorS8OObDv3HH2cvaxO2B1htyZVG-I_u-VLkPsKB54zh=w1920-h948-rw)

__Loading the datasert__  
The path of the dataset (Train and Test subdirectories) was fed into
‘load_dataset’ function that returns a dictionary containing the list of folder names (the
category names) as ‘target’, and list of all the individual file names as ‘filenames’.  
__One hot Encoding__  
The target categories produced by the load_dataset were then one hot encoded to 1d arrays of size 12 as y_train and y_test 
__Train-Validation Split__
The train part of the dataset was further split in the ratio of 80:20 to create a validation set only for Log Reg Models was this validation split done in case of the dense net (benchmark model) the validation split was specified in the train_datagen_vgg16.  
__Defining the datagenerators__  
Since the images are of plants and the subject is in center only simple augmentation was performed
the code for the generator is as follows   
![Generator Code ](https://lh6.googleusercontent.com/7bYS94xmisH0ARoai3LmCwAlNjI3a1GEmSu5E1mkSWw8tiSn95A0TPJ-7J-PD9yhsD8KJZBmS4X3pgGpOvXM=w1920-h948-rw)

### Implementation
In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_  

__The Benchmark Model (VGG16 with only 2 trainable dense layers at the top)__  
Following are the key properties   
- Tensors of shape (224,224,3) representing the image shape and 3 channels were
fed into the network
- All the layers of the pre-trained VGG16 were frozen.
- Additionally, for predictions, the output of the convolution part was flattened and was fed into a dense net with 2 layers  
- The 1st layer was of 1024 nodes and had the relu activation function and a dropout of 0.5
- The 2nd layer was of 12 nodes and had a sigmoid activation function 
- The architecture resulted in 25,703,436 Trainable Parameters  
The summary of the model can be found in the figure above 

__For Logistic Regression Classifiers trained using the Bottleneck Features__  
The idea here was to test the performance of different models as feature extractors.  
I ended up using VGG16, Xception and MobilNet to represent models of different sizes refer [link](https://keras.io/applications/) for more details
- First, the input images were loaded and reshaped to required input shape depending on the model used as the feature extractor the target size was 224x224 for VGG16 and MobilNet and 299x299 for Xception
- Then the model in question was loaded with the weights as imagenet, and pooling set to avg
- The train test and validation images were then run through the predict function to obtain the bottleneck features 
- THese Bottleneck features were then saved locally and used to train a Logistic Regression model
with the parameters as multi_class='multinomial' and solver='lbfgs'

__Complications__  
Although the same methodology was used in the case of all the 3 models the Xception models features were unable to converge even after 20000 iterations.  
_The mentor told me to skip the Xception model as the results may not converge without in its case_  
Since I didn't want to change the pre-trained layers of Xception. As it would be unfair for the other models in comparison. I left the results be and proceded with the MobileNet Features for refinement
The results are further explained in the Results 


### Refinement

As the Bottleneck features of MobileNet gave the best result, they were the ones selected for refinement 
the model to be optimized was the Logistic Regression model
The hyperparameters for a Logistic Regression model are C and penalty 
the default value of which are 1 and 12 respectively the solver used only supports the penalty of 12 so 
I implemented gridsearchCV on the hyperparameter C to further tune the model  
The code was as follows
![Grid_search](https://lh6.googleusercontent.com/7JKaMmCkYWqOK_xBkYx52NLMTYdlt-mOLnm2ydbx6vdMR7x6hfVz0XoMMH_p1ZsXkuqAEC6w0kyIBu_4ceqM=w1920-h948-rw)  

The initial F1 score obtained was 0.8171 with no training of the MobileNet on the data just using the default ImageNet weights.
after refinement i.e GridSearchCV, the score obtained was 0.8474
The Grid search used the default 3 fold CV for finding the optimal parameters 

## IV. Results

### Model Evaluation and Validation
The Overall Results of the various combination tested is  summarized in the table below 
The data used for calculating the F1 Score is Validation split of the data
| Model                                                         | F1 Score(Validation Data) |
|---------------------------------------------------------------|:--------:|
| Using VGG16 Bottleneck Features and Logistic Regression       | 0.7961   |
| Using Xception Bottleneck Features and Logistic Regression    | 0.425    |
| Using MobileNets Bottleneck Features and Logistic Regression  | 0.811    |
|Using MobileNets Bottleneck Features and Logistic Regression (Optimized)  | 0.8471    |


*NOTE: Benchmark models results are not included here and are reported later when testing data s used for final comparison*

The final result is just as I expected. I wanted to approach the Problem with scale and speed in mind in return I had to give up performance and the goal was to minimize the performance loss as much as possible.

Although the VGG16 or InceptionV3 (as used in many Kaggle Kernels) were giving better metrics, these models are very memory heavy and difficult to run. To remedy this I used the MobileNet Model which is very lightweight and used Logistic regression instead of a dense Net for classification to improve speed.

The Initial version of the Model using MobileNets Bottleneck was further optimized using gridsearchCV. On testing the model’s performance on the unseen dataset – the test dataset, with the
model obtained from the training process, the classifier scored an F1 score of 0.8474. 

The model built is robust as its built with Neural Nets which by there own nature account for variances.  
How this score was achieved is discussed in the ‘Refinement section’ and how can I further
improve this score is discussed in the ‘Improvement’ section. 
The justification for these results is provided below in the Justification section.

### Justification
The initial Benchmark model used VGG16 with Image Augmentation and achieved an F1 Score of 0.811 on the testing set.

The final model that I selected used Mobilnet without any Image Augmentation for Feature Extraction and A Logistic Regression model for Classification and achieved an F1 score of 0.8474 on the testing set after optimizing using GridsearchCV for C which was later found to be 1 itself.

Which is better than the Benchmark Model but is much Faster to train and Run. This achieves the end goal to achieve similar or better accuracy with a faster and simpler architecture. 

I believe the final solution will definitely contribute significantly towards solving the current problem in a real-world setting (i.e with computation speed in mind) 




## V. Conclusion

### Free-Form Visualization
I decided to plot the confusion matrix of the final classifier to get a better picture of the models' performance and to observe which category is poorly classified by the classifier
![Confusion Matrix](https://lh6.googleusercontent.com/yo3UY3fyHVr7CcqBT2MjzLmL8ghvQM0Do4UQVfKHxjRE4bNgHHikVBqqD9zjieIEGt93RvF3JljCrL-oY53e=w1920-h948-rw)

We can see in the confusion matrix above, that the major misclassification happened between Loose Silky-bent and Black-grass.
The classifier is having difficulty distinguishing these two classes apart out of the 125 samples of  Loose Silky-bent 14 were marked as Black Grass and 7 as Common Wheat which is also significant.

The situation is much worse in case of Black Grass out of the 43 samples 22 were marked as Loose Silky bent. The Imbalanced nature of the dataset can be seen here

This is were the classifire needs improvement as this amounts to the majority of the miss classifications  

### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

After my experience at the Google Indian Hackathon, where the winner was the MobileNet Architecture for the image classification problem where the target device was a smartphone. I wanted to better my understanding at Image Classification especially when there are not a lot of computation resources available (Mobile Devices) and the classification needs to be fast. I believe that the choice of this problem has justified that need. 

For this problem, after downloading the dataset from Kaggle, As the data was just in one subdirectory of train
I renamed it to train 1 and split the data into two parts in 80:20 ratio and created a training and testing split.
This was done so that Image augmentation could be performed on the data.

I loaded the dataset using scikit learns load_files() function and converted the categorical file names into an array of 1s and 0s using the one-hot encoding (to_categorical) Function.  
I explored the data by visualizing the categorical distribution of the dataset by plotting a graph to check if the dataset is well balanced or not. I further plotted randomly picked samples for each of the categories to understand
the image samples. 
 
After that, I performed image augmentation on the training and the validation data and used transfer learning with the pre-trained VGG16 model on Imagenet and then to allow training the dense net at the top of the model I converted the paths of the images to 4d tensors of appropriate dimensions to VGG16s input. 
The number of epochs were initially set to 50 and then the model was rerun for another 50 epochs but stopped early after completing 13 epoch.    

After training and evaluation of the before mentioned benchmark model. I loaded the models of VGG16, Xception and MobileNet models without the top and average pooling at the end to extract the bottleneck features of the images.
These were stored locally and they were used to train a separate Logistic Regression model.
The maximum iteration parameters was increased to 5000 so that the classifier converges (Leaving the Xception model which did not converge even after 20000 iterations)
    
Evaluating the performance of the classifier on the validation set I took the mobilents bottleneck features for the final model as they gave decent performance with the fastest prediction speed. I implemented GridsearchCV to tune the hyperparameter C and find out the optimal value to be 1.  

Selecting this optimized classifier as the final model I evaluated its performance and to take deeper look plotted the confusion matrix for the testing dataset of 960 samples.  


Working on this problem has taught me a lot although not implemented I learned about masking. I learned about Image Augmentation and transfer learning. I also explored how CNN's can be used in conjunction with the traditional Machine learning models.  
This project gave me a good insight on how to deal with future image classification problems and encouraged me to work on further improving my current model especially try out masking the background and to use augmentation to balance the dataset.
It is satisfying to see that the final model performs exceptionally well and I can't wait to work on other projects
   

### Improvement

As an improvement and future work the following can be done
- I would like to try data masking on the training set.  
Noise from the background of the images (especially from the barcodes) can be canceled by masking images. I believe
that without the background noise and restricting the visibility to the leaves, the
model can be trained better, and we may notice a significant improvement in the
performance.  

- Another implementation that can be tried is data augmentation. As the dataset is highly
unbalanced, augmenting data to the under-represented classes might give a good boost
to the total number of training images yielding a well-balanced dataset. Training the
model on such dataset may give us a significant improvement in the performance. 
- Another improvement that comes to mind is using a different model than Logistic Regression like a Random Forest which may result in improvement.

-----------

### References
- https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks/
- http://cs231n.github.io/convolutional-networks/
- https://keras.io/layers/convolutional/ 
- https://keras.io/layers/core/
- https://engmrk.com/convolutional-neural-network-3/?utm_campaign=News&utm_medium=Community&utm_source=DataCamp.com
- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html



