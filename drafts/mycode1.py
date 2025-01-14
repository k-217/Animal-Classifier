# 17.565% on test
# add more layers (or use a pre-trained model) and train longer
# include predicate matrix

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime

from torchvision import datasets, transforms, models

import torch

from PIL import Image

''' Step 1: Define the path to the training and test directories '''

train_dir = "train"
test_dir = "test"

transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation = 2, max_size = None, antialias = True),
    transforms.ToTensor(), # converts a PIL Image ot pytorch tensors
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]), # channel-wise means and standard deviations for RGB (used for ImageNet-pretrained models).
    transforms.RandomHorizontalFlip() # Randomly flips some images   
    # can have more transforms here for data augmentation and improving generalization
])

BATCH_SIZE = 32 # mini-batch gradient descent

print("[INFO]: Loading animal images dataset...")

# Dataset: stores and manages data
animal_dataset = datasets.ImageFolder(root = train_dir, transform = transform) 

# (80% train, 20% validation)
train_size = int(0.8 * len(animal_dataset))
val_size = len(animal_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(animal_dataset, [train_size, val_size], generator = torch.Generator().manual_seed(42))

# DataLoader: manages how data is fitted into model
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = False) 

trainSteps = len(train_loader.dataset) // BATCH_SIZE
valSteps = len(val_loader.dataset) // BATCH_SIZE

# can also have classes for dataset and dataloader!

''' Step 2: Define a model architecture for image classification. '''

class AnimalClassifier(torch.nn.Module):  # nn.Model: Base class for all neural network modules
    
    def __init__(self, input, num_classes) -> None:

        super(AnimalClassifier, self).__init__()

        # Using here a sequential (linear) convolutional network,
        # Instead, may define different layers and connect in forward().

        self.cnn = torch.nn.Sequential(

            # 5 sets of CONV => RELU => POOL layers

            torch.nn.Conv2d(in_channels = 3, out_channels = 8, kernel_size = 3, padding = 1), # 3 RGB channels, 8 filters, 3x3 kernel, padding to keep size
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = (2, 2), stride = 2), # 2x2 pool

            torch.nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3, padding = 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = (2, 2), stride = 2),

            torch.nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = (2, 2), stride = 2),

            torch.nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = (2, 2), stride = 2),

            torch.nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = (2, 2), stride = 2)
        )

        # find a way to incorporate the predicate matrix!

        # initialize weights.

        self.fc = torch.nn.Sequential(

            torch.nn.Linear(in_features = 128 * 6 * 6, out_features = 512), # find a function to avoid hardcoding in_features
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5), # During training, randomly zeroes some of the elements of the input tensor 
            torch.nn.Linear(512, num_classes) 

        )

    def forward(self, x):
        
        x = self.cnn(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
def init_weights(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

''' Step 3: Train the model. '''

INPUT_SIZE = 224 * 224 * 3  # Flattened size of input images
NUM_CLASSES = len(animal_dataset.classes)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-4
WEIGHT = None # Tensor of size NUM_CLASSES and floating point dtype
REDUCTION = 'mean' # 'none' | 'mean' | 'sum'
LABEL_SMOOTHING = 0.0
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 10
H = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": []
}

print("[INFO]: Initialising the AnimalClassifier model...")

# model
model = AnimalClassifier(INPUT_SIZE, NUM_CLASSES).to(DEVICE)
model.apply(init_weights)

# loss function: CrossEntropyLoss = Negative Log Likelihood Loss + Log Softmax
lossfn = torch.nn.CrossEntropyLoss(weight = WEIGHT, reduction = REDUCTION, label_smoothing = LABEL_SMOOTHING)

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)

# schedulers to update learning rate after each epoch
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)

print("[INFO]: Training the network...")

start_time = datetime.now()

# learning alagorithm: backpropagation

for epoch in range(NUM_EPOCHS):

    # set model in "train" mode
    model.train()

    total_train_loss = 0
    train_correct = 0

    total_val_loss = 0
    val_correct = 0

    for images, labels in train_loader:

        # send input to "device"
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # images = images.view(images.size(0), -1) 

        # reshapes the images tensor so that it becomes a 2D tensor with rows = BATCH_SIZE and cols = total number of pixels
        # -1 ensures that the cols are automatically calculated

        outputs = model(images)

        loss = lossfn(outputs, labels)

        optimizer.zero_grad() # setting all gradients to zero
        loss.backward() # computing gradients 
        optimizer.step() # updating weights

        total_train_loss += loss.item()
        train_correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()

    train_loss = total_train_loss/len(train_loader.dataset)
    train_accuracy = train_correct/len(train_loader.dataset)

    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] - Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")
    
    # evaluating with validation dataset

    with torch.no_grad():

        # set model in "eval" mode
        model.eval()

        for images, labels in val_loader:

            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)

            loss = lossfn(outputs, labels)

            total_val_loss += loss.item()
            val_correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()

        val_loss = total_val_loss/len(val_loader.dataset)
        val_accuracy = val_correct/len(val_loader.dataset)

        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] - Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    H['train_loss'].append(train_loss)
    H['train_accuracy'].append(train_accuracy)
    H['val_loss'].append(val_loss)
    H['val_accuracy'].append(val_accuracy)

    scheduler.step()

# from sklearn.metrics import confusion_matrix
# y_true = [label.item() for _, label in val_loader]
# y_pred = [torch.argmax(model(img.to(DEVICE)), 1).item() for img, _ in val_loader]
# cm = confusion_matrix(y_true, y_pred)
# print(cm)

end_time = datetime.now()
time_taken = end_time - start_time

print(f"[INFO]: Completed training in {time_taken.seconds} seconds and {time_taken.microseconds} microseconds!")

''' Step 4: Visualize the loss and accuracy on a graph. '''

plt.style.use('ggplot')
plt.figure()
plt.plot(H["train_loss"], label = "train_loss")
plt.plot(H["val_loss"], label = "val_loss")
plt.plot(H["train_accuracy"], label = "train_accuracy")
plt.plot(H["val_accuracy"], label = "val_accuracy")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc = "lower left")
plt.show()


''' Step 5: Test the model. '''

with torch.no_grad():

    # set model in evaluation mode
    model.eval()

    test_images = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
    test_predictions = []

    for img_name in test_images:
        
        img_path = os.path.join(test_dir, img_name)
        
        image = Image.open(img_path).convert('RGB')
        
        image = transform(image).unsqueeze(0).to(DEVICE)

        outputs = model(image)
        predicted_class = torch.argmax(outputs, dim = 1).item()
        test_predictions.append((img_name, animal_dataset.classes[predicted_class]))


'''Step 6: Save predictions to a CSV file'''

submission = pd.DataFrame(test_predictions, columns = ['image_id', 'class'])
submission.to_csv("predictions.csv", index = False)