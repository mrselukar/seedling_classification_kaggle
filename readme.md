### Proposal Review Link  
https://review.udacity.com/#!/reviews/1610899  

### Capstione Review Link
https://review.udacity.com/#!/reviews/1633836

### Drive Link for the restructured dataset  
https://drive.google.com/open?id=1Uwtv81b4RQC1nm1o4dti11yabdFW3JK_    

### Original Datasets and Inputs  
The dataset is hosted on [Kaggle](www.kaggle.com) and is free to download. 
The data is provided by The Aarhus University Signal Processing group, in collaboration with the University of Southern Denmark  
Please visit this [link](https://www.kaggle.com/c/plant-seedlings-classification/data) to get the data via Kaggle.  

#### Regarding the Xceptions poor performance  
I asked the same question to my mentor on slack and the conclusion was that the Xceptions Bottleneck features wont result in a convergence in 
case of Logistic Regression as it is. Since I did not wish to change the parameters as it would be unfair to other models in the comparison 
I let the result be.

### Software/Libraries   
I used Google Colab and did not require to install any extra libraries  

### Bottleneck Features and Saved Weights for the benchmark model can be found here  
https://drive.google.com/open?id=1wJbzHyD39ADQG_xONv1mfM2q5l5R36jN    

For splitting always use the seed as 42 and the ops will obtained   
I realized that later that the ground truth should also be stored  

The model weights are stored as VGG16_1_1.h5
