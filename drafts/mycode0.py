import os
import pandas as pd
import numpy as np

from torchvision import datasets, transforms, models

import torch

from PIL import Image

''' Step 1: Define the path to the training and test directories '''

train_dir = "train"
test_dir = "test"

transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation = 2, max_size = None, antialias = True),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]), # channel-wise means and standard deviations for RGB (used for ImageNet-pretrained models).
    # can have more transforms here for data augmentation and improving generalization
])

BATCH_SIZE = 32 # mini-batch gradient descent

train_dataset = datasets.ImageFolder(root = train_dir, transform = transform) # stores and manages data
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True) # manages how data is fitted into model

# what is the use of classes.txt file?

# 1. Utilities associated with the dataset:
# print(f"Number of samples: {len(train_dataset)}")
# print(f"Number of classes: {len(train_dataset.classes)}")
# print(f"Classes: {train_dataset.classes}")
# print(f"Class to Index Mapping: {train_dataset.class_to_idx}")
# print(f"Image shape: {train_dataset[0][0].shape}")
# print(f"Image size: {train_dataset[0][0].size}")
# print(f"Image mode: {train_dataset[0][0].mode}")
# print(f"Image mean: {np.mean(np.array(train_dataset[0][0]))}")
# print(f"Image std: {np.std(np.array(train_dataset[0][0]))}")
# print(f"Image min: {np.min(np.array(train_dataset[0][0]))}")
# print(f"Image max: {np.max(np.array(train_dataset[0][0]))}")
# print(f"Image dtype: {np.array(train_dataset[0][0]).dtype}")

# 2. Utilities associated with the dataloader:

# print(f"Number of batches: {len(train_loader)}")
# print(f"Batch size: {train_loader.batch_size}")
# print(f"Number of samples: {len(train_loader.dataset)}")
# print(f"Drop last: {train_loader.drop_last}")
# print(f"Batch sampler: {train_loader.batch_sampler}")
# print(f"Sampler: {train_loader.sampler}")
# for imgs, labels in train_loader:
#         print(f"  Images shape: {imgs.shape}, Labels: {labels.shape}")
#         break





# check convolutional layers in torch documentations to build a neural network model

# also, vision layers specially for images

# also, pooling: operation in cnns, helping reduce the spatial dimensions of feature maps while retaining important information

# non-linear activation functions: 
# - nn.ELU: Exponential Linear Unit
# - nn.ReLU: Rectified Linear Unit
# - nn.sigmoid: Sigmoid function
# - nn.Tanh: Hyperbolic Tangent
# - nn.softplus: Softplus function
# - see other functions in the documentation

# regularization: reduces overfitting
# Lasso: (Least Absolute Shrinkage and Selection Operator) L1 regularization: adds the “absolute value of magnitude” of the coefficient as a penalty term to the loss function(L)
# *Ridge:  adds the “squared magnitude” of the coefficient as a penalty term to the loss function(L).
# Elastic Net: combines both L1 and L2 regularization

# L1 regularization prefers to put weights on one parameter and L2 spreads the weight.

# dropout (a method of regularization): 
# let say d3 is the droupout layer for activation layer a3 with a keep probability of 0.8, then the output tensor will have 20% of its elements zeroed out
# next, a3 = np.multiply(a3, d3) # element-wise multiplication and then, a3 = np.divide(a3, 0.8) # scaling
# this is also done to prevent overfitting

# other regularization techniques: i) data augmentation (rotating, flipping, colour jitter images)

# early stopping

# orthogonalization: separate the optimization of the loss function and the regularization of the model

# batch normalization

# normalization layers: check documentation

# distance functions: 
# nn.CosineSimilarity(dim = 1, eps = 1e-8) and 
# nn.PairwiseDistance(p = 2, eps = 1e-6, keepdim = False) # p: norm degree, eps: small value to avoid division by zero, keepdim: whether to retain the last dimension in the output tensor

# can also make a class for the dataset

# class AnimalDataset(torch.nn.Dataset):
#     def __init__(self, image_paths, labels, aux_features, transform=None):
#         """
#         :param image_paths: List of paths to image files.
#         :param labels: List of class labels for each image (0 to 49).
#         :param aux_features: Numpy array or Tensor of shape (50, 85) containing class-level features.
#         :param transform: Torchvision transforms to apply to images.
#         """
#         self.image_paths = image_paths
#         self.labels = labels
#         self.aux_features = aux_features
#         self.transform = transform

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         # Load and preprocess the image
#         image = Image.open(self.image_paths[idx]).convert("RGB")
#         if self.transform:
#             image = self.transform(image)

#         # Get the label and corresponding auxiliary features
#         label = self.labels[idx]
#         aux_feature = self.aux_features[label]  # Fetch auxiliary features for the class

#         return image, torch.tensor(aux_feature, dtype=torch.float32), label


''' Step 2: Define a neural network for image classification '''

# convolutional neural networks (CNNs): used for image classification, object detection, and segmentation tasks

# 3x3 filter, 9 parameters: train them! 
# nxn image and fxf filter -> (n-f+1)x(n-f+1) feature map
# to avoid losing the corners of image and shrinking of image while several layers of convolution, padding is used
# valid and same convolutions
# strided convolutions
# convolution over volume (RGB images: 3 channels): no of channels in filter = no of channels in image: 2 filters and then apply non-linearity changing channels to 2 (this is one layer, can have as many filters in layers: so is the number of parameters to learn)

# pooling layers: takes a maximum or average value of a region of the image, no learning parameters: hyperparameter: filter and stride

# fully connected layers: flatten the output of the last convolutional layer and pass it through a series of fully connected layers

# kernel size, no of filters, padding, stride, bias

# nearest neighbour
# 




class SimpleFeedForwardClassifier(torch.nn.Module):

    # nn.Model: Base class for all neural network modules


    def __init__(self, input_size, num_classes) -> None:
        super(SimpleFeedForwardClassifier, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.fc = torch.nn.Linear(input_size, num_classes)
        # define convolutional neural layers here

        # example:
        # self.cnn = models.resnet50(pretrained = True) # pretrained model to extract image features
        # self.cnn.fc = torch.nn.Identity()
        # self.aux_processor = torch.nn.Sequential( # Auxiliary feature extractor
        #     torch.nn.Linear(85, 128),
        #     torch.nn.ReLU(),
        #     torch.nn.Dropout(0.3),
        #     torch.nn.Linear(128, 64),
        #     torch.nn.ReLU()
        # ) 
        # self.fc = torch.nn.Sequential( # fusion and classification layer
        #     torch.nn.Linear(64 + 2048, 256),  # 2048 from ResNet50, 64 from aux_processor
        #     torch.nn.ReLU(),
        #     torch.nn.Dropout(0.5),
        #     torch.nn.Linear(256, num_classes)
        # )

    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)
    
        # Process images through CNN
        image_features = self.cnn(images)
        
        # Process auxiliary features
        aux_features = self.aux_processor(aux_features)
        
        # Combine features
        combined = torch.cat([image_features, aux_features], dim=1)
        
        # Final classification
        return self.fc(combined)
    
    # can also contain some buffer tensors that do not require grads and hence, are not registered as parameters of the model

# Option1: Use a pretrained model like ResNet, VGG, or EfficientNet (from torchvision.models) and fine tune it with dataset. 
# Option2: Freeze the earlier layers of the pretrained model and fine-tune only the later layers.

''' Step 3: Define the training setup '''

INPUT_SIZE = 224 * 224 * 3  # Flattened size of input images
NUM_CLASSES = len(train_dataset.classes)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-2
WEIGHT = None # Tensor of size NUM_CLASSES and floating point dtype
REDUCTION = 'mean' # 'none' | 'mean' | 'sum'
LABEL_SMOOTHING = 0.0

# model
model = SimpleFeedForwardClassifier(INPUT_SIZE, NUM_CLASSES).to(DEVICE)

# m = torch.nn.Dropout(p = 0.5, inplace = False)
# x = torch.randn(10, 10)
# y = m(x) # applies dropout to the input tensor x, i.e. randomly zeroes some of the elements of the input tensor with probability p
# # effective technique for regularization and preventing the co-adaptation of neuron
# # Furthermore, the outputs are scaled by a factor of 1/(1-p) during training. This means that during evaluation the module simply computes an identity function.

# To reset all the gradients of the model parameters to zero, use: model.zero_grad(set_to_None = True

# loss function
criterion = torch.nn.CrossEntropyLoss(weight = WEIGHT, reduction = REDUCTION, label_smoothing = LABEL_SMOOTHING)
# LogSoftMax + NLLLoss (negative log likelihod) = CrossEntropyLoss (used for multi-class classification problems)
# Multinomial Logistic Regression

# or, could be:

# nn.MSELoss: Mean Squared Error Loss
# nn.L1Loss: Mean Absolute Error Loss
# nn.SmoothL1Loss: Huber Loss
# nn.NLLLoss: Negative Log Likelihood Loss
# nn.PoissonNLLLoss: Poisson Negative Log Likelihood Loss
# nn.GaussianNLLLoss: Gaussian Negative Log Likelihood Loss
# nn.KLDivLoss: Kullback-Leibler Divergence Loss
# nn.BCELoss: Binary Cross Entropy Loss
# nn.HuberLoss: Huber Loss
# nn.CosineEmbeddingLoss: Cosine Embedding Loss
# nn.MultiMarginLoss: Multi-Margin Loss
# check more on documentation and see which works here

# Multi-Class SVM Loss

optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

# Adam: 
# SGD: Stochastic gradient descent
# Adadelta, Adafactor, Adagrad, etc. (check documentation)

# optim.Optimizer: Base class for all optimizers

# To update learning rate of optimizer after each epoch:
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# scheduler = ExponentialLR(optimizer, gamma=0.9)
# scheduler2 = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
# see all schedulers in documentation
# for epoch in range(20):
#     for input, target in dataset:
#         optimizer.zero_grad()
#         output = model(input)
#         loss = loss_fn(output, target)
#         loss.backward()
#         optimizer.step()
#     scheduler.step()
#     scheduler2.step()

# To reset all the gradients of all the optimized tensors to zero, use: optimizer.zero_grad(set_to_None = True)

# Stochastic Weight Average (SWA): see about this
# Exponential Moving Average (EMA): see about this

# prune the model: removes non-essential parts of a model to reduce its complexity and size while improving its performance
# torch.nn.utils.prune

NUM_EPOCHS = 3

''' Step 4: Train the model '''

model.train()

# learning algorithm: backpropagation

# discriminative or generative?

for epoch in range(NUM_EPOCHS):

    # backpropagation
    
    total_loss = 0
    
    for images, labels in train_loader:
        
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        images = images.view(images.size(0), -1)
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        # see autograd and methods of implementation
        optimizer.zero_grad()
        loss.backward() # backward propagation
        optimizer.step() # forward propagation

        total_loss += loss.item()

        # Early Stopping: stop when the performance starts degrading or stop before the model starts memorizing the data, i.e. may overift.
        # when?
        # make graphs? (of the training and test both)
    
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {total_loss/len(train_loader):.4f}")





''' Step 5: Test the model on the test set '''

# attention mechanisms and gating (by adjusting weights)

model.eval()

test_images = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
test_predictions = []

for img_name in test_images:
    
    img_path = os.path.join(test_dir, img_name)
    
    image = Image.open(img_path).convert('RGB')
    
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        predicted_class = torch.argmax(outputs, dim = 1).item()
        test_predictions.append((img_name, train_dataset.classes[predicted_class]))

# accuracy, precision, recall, and F1 Score: metrics to evaluate



'''Step 6: Save predictions to a CSV file'''

submission = pd.DataFrame(test_predictions, columns = ['image_id', 'class'])
submission.to_csv("predictions.csv", index = False)
