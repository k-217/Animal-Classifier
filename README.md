# Animal-Classifier

## Introduction

This project is a basic neural network setup for image classification into classes of animals. Though it attempts to mimic how a human observes and infers, the simple rule behind it is to decipher the images into mathematical patterns and connect them to known categories and traits. The project focuses on building the model from scratch without pretrained architectures to reinforce foundational understanding of convolutional neural networks (CNNs). Major focus was put on improving its limited understanding from the meaningful features from images and further training it in a way that it interprets context beyond simply image patterns to the semantic features.

## Python Utilities

- pytorch: to load datasets, define model architecture, train the model, and test the model.
- matplotlib: to plot model loss and accuracy
- pandas: to save predictions of model to a csv file
- PIL (Python Imaging Library): to load images
- tqdm: to log the epoch progress with accuracy and loss updates
- os: to handle directory structures

## Data

### Raw data
- Dataset: Contains over 10,000 images spanning 50 animal classes, divided into train and test folders.
- predicate-matrix-binary.txt: Binary matrix representing 85 semantic features for each class.
- classes.txt: A file listing all class labels.

### Data Preprocessing

- Addressed discrepancies in the dataset:
    - Training folder contains only 40 classes, while the test folder includes 50 (unlabeled) classes.
    - Updated classes.txt and predicate-matrix-binary.txt to align known classes and account for unseen ones.

- Split the training data into 80-20 ratio for training and validation. 

- Randomly grouped into batches of 32 for applying mini-batch gradient descent later for the training.

- Image Preprocessing:
	- Resize(): resizes images to a standard size of 224x224 pixels using bilinear interpolation.
	- ToTensor(): converts PIL image into tensor with pixel values scaled to [0, 1]
	- Normalize(): normalizes the images using channel-wise mean and standard deviations (RGB channels), consistent with ImageNet standards
		- Mean: [0.485, 0.456, 0.406]
		- Standard Deviation: [0.229, 0.224, 0.225]
		- Normalized Pixel Value $= \frac{Pixel Value - Mean}{Standard Deviation}$


### Data Augmentation
- RandomHorizontalFlip(): randomly flips some of the images in the batches
- ColorJitter(): randomly changes image brightness, contrast, saturation, and hue.
- RandomAffine(): applies random affine transformations such as rotation, scaling, and translation

## Model Architecture

- A multi-task learning approach (or, auxiliary task learning) to the classification problem that learns both
    - the image features (Pytorch tensor of its pixel values): single-label, and
    - the predicate features (semantics with the picture): multi-label

- Five convolutional layers (applied to image tensor): 
    - Convolution
    - Batch Normalization: to stabilize learning
    - ReLU Activation: to introduce non-linearity to the model
    - Maximum Pooling: to reduce the dimensionality of the images
- Fully connected layer for classification by image features
    - With ReLU activation and Dropout for generalization
- Fully connected layer for predicate (to the same image tensor after convolution)
    - 2 x (ReLU and Dropout) and then converting to a linear with num_predicates attributes to compare with the predicates

- Returns both the predicates and class label for the training and other computations

## Training Setup

- Weight and Bias Initialization
    - weights are set by kaiming_normal ensuring that the variance of activations is maintained across layers, avoiding vanishing or exploding gradients during training 
    - all bias is set to zero

- Optimizer
    - Adam Optimizer: Adaptive Moment Estimation, a combination of the ‘gradient descent with momentum’ algorithm and the ‘RMSP’ algorithm.
    - Learning Rate = 1e-3
    - Weight Decay = 1e-4

- Learning Rate Scheduler
    - OneCycleLR(): anneals the learning rate from an initial learning rate to some maximum learning rate and then from that maximum learning rate to some minimum learning rate much lower than the initial learning rate.

- Loss Functions
    - CrossEntropyLoss(): for the classes labels
        - CrossEntropyLoss(reduction = 'mean') $= -\frac{w_{y_i}*\log(\frac{\exp(x_{i, y_i})}{\sum_{c=1}^C (\exp(x_{i, c}))})}{\sum_{i=1}^N (w_{y_i})}$
        - (Similar to applying Softmax and then Negative Log Likelihood Loss)
    
    - BinaryCrossEntropyLoss(): for the predicates predicted by the model
        - $BCE = -w_i*\frac{1}{N}\sum_{i=1}^N (y_i*\log(p(y_i))+(1-y_i)*\log(1-p(y_i)))$
    
    - lossfn_alignment() (Custom loss function): for the alignment of image features predicted and the predicates of the class. 
        - takes dot product of predicted class tensor from model with the predicate matrix, and
		- computes mse loss with this projection and the true predicates from the matrix

- Number of Epochs
    - 30 

- Hardware
    - Kaggle GPU P100

## Testing Setup

- utilizes a simpler zero-shot learning approach

    - calculate probability of a class by softmax on the classes returned by the model, max = predicted_class1

    - take a dynamically weighted similarity: Intersection over Union and Cosine Similarity to check the predicates

    - the class with higher similarity = predicted_class2

    - if the predicted_class2 is in the unseen bunch, actual predicted class = predicted_class2

    - if not,
        - if the probability of that class is greater than a confidence threshold, actual predicted class = predicted_class2
	    - if not, actual predicted class = predicted_class1

    - return the actual predicted class

## Results

- Training Accuracy: ~70%

- Validation and Test Accuracy: ~30-40%

## Process and Challenges Encountered

- Initial implementation resulted in low accuracy (~10% on the test set). Subsequent iterations improved results through:
    - Data augmentation techniques.
    - Optimizer, scheduler, and loss function refinements.

- A pivotal change involved predicting predicates separately and introducing alignment loss, which improved validation accuracy to ~40%.

- Challenges faced:
    - Proper integration of the predicate matrix.
    - Computational overhead due to added complexity.
    - Limited understanding of confusion matrices and class-specific optimizations.

- Future directions include refining hyperparameters, exploring pretrained models, and explicitly implementing zero-shot learning.

## Explainability Report

In the way how a human would identify animals, the network followed this method for understanding images:
- Edges and textures: The first layer identifies simple patterns like edges or corners, similar to how humans see outlines.

- Complex structures: As the image passes through deeper layers, the model combines simple patterns to detect more complex features like a beak, claws, or fur patterns.

- After extracting features, the model condenses the information into the most noticeable features while ignoring the unnecessary details (e.g., a lion's mane or an elephant’s trunk), using pooling layers.

The performance of the model indicates that it does extract meaningful features for seen classes. These features likely include:
- Distinct patterns like stripes for zebras.
- Shape-based traits like trunks for elephants.
- Texture-related features like fur or scales.

The model achieves reasonable performance on unseen classes due to the semantic predicates (binary matrix) guiding the predictions. 
- The predicate “stripes” helps identify unseen animals like okapis or other striped species.

- The predicate “flys” aids in distinguishing birds from terrestrial animals.

The accuracy gap highlights that the predicates and image features must be perfectly aligned for optimal performance, leaving a scope for future development.


## Learning Outcomes

- Basic implementation of a neural network

- Mathematics behind a deep learning model and the training algorithm: backpropagation

- Tradeoff between accuracy, complexity, and computational overhead: fine-tuning the parameters, iterative problem solving techniques

- Zero-shot learning concepts

## Conclusion

While the model demonstrates low/moderate accuracy, it underscores the importance of experimentation and theoretical grounding in building custom architectures. Model performance is limited by hyperparameter choices and the absence of pretrained architectures. Future work could involve hyperparameter tuning, incorporating PCA, leveraging embeddings for semantic relationships, and explicitly implementing zero-shot learning.

## Reference

This is a submission to the Pixel Play Challenge by Vision and Language Group, IIT Roorkee. The problem statement and referenced data may be found on [Kaggle](https://www.kaggle.com/competitions/vlg-recruitment-24-challenge/). 

## Info

the ```drafts``` folder contains previous codes that were improvised to get the final one. Feel free to ignore them.
