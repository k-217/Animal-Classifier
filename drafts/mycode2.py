# ~10% accuracy on test dataset
# though worked perfect on the validation dataset
# think about the predicate matrix for test data instead of simply -1s

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
predicate_matrix_dir = "predicate-matrix-binary.txt"

transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation = 2, max_size = None, antialias = True),
    transforms.ToTensor(), # converts a PIL Image to pytorch tensors
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

print("[INFO]: Loading predicate-matrix-binary.txt...")

with open(predicate_matrix_dir) as f:
    lines = f.readlines()

predicate_matrix = []

for line in lines:
    values = line.strip().split()
    values = [float(i) for i in values]
    predicate_matrix.append(values)

predicate_matrix = torch.tensor(predicate_matrix, dtype = torch.float32)

''' Step 2: Define a model architecture for image classification. '''

print("[INFO]: Defining the model architecture...")

class AnimalClassifier(torch.nn.Module):  # nn.Model: Base class for all neural network modules

    def __init__(self, input, num_classes, num_predicates) -> None:

        super(AnimalClassifier, self).__init__()

        # Using here a sequential (linear) convolutional network,
        # Instead, may define different layers and connect in forward().

        self.cnn = torch.nn.Sequential(

            # 5 sets of CONV => RELU => POOL layers

            # 3 RGB channels (in-channels), 16 filters (out_channels), 3x3 kernel, padding to keep size

            # initial: 224x224x3

            torch.nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, padding = 1), # 224x224x16
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = (2, 2), stride = 2), # 112x112x16

            torch.nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 1), # 112x112x32
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = (2, 2), stride = 2), # 56x56x32

            torch.nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1), # 56x56x64
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = (2, 2), stride = 2), # 28x28x64

            torch.nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1), # 28x28x128
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = (2, 2), stride = 2), # 14x14x128

            torch.nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, padding = 1), # 14x14x256
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = (2, 2), stride = 2) # 7x7x256
        )

        # [32, 3, 3, 256]

        # predicate matrix
        self.predicate_fc = torch.nn.Sequential(
            torch.nn.Linear(in_features = num_predicates, out_features = 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5)
        )
        
        
        self._dummy_input = torch.zeros(1, 3, 224, 224)  # Use a dummy input to get the size
        self._cnn_output_size = self._get_cnn_output_size(self._dummy_input)  # Get size after CNN

        self.fc = torch.nn.Sequential(

            torch.nn.Linear(in_features = self._cnn_output_size + 128, out_features = 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5), # During training, randomly zeroes some of the elements of the input tensor 
            torch.nn.Linear(512, num_classes) 

        )

    def _get_cnn_output_size(self, x):
            with torch.no_grad():
                output = self.cnn(x)
                return output.size(1) * output.size(2) * output.size(3)

    def forward(self, x, predicate):
        
        output = self.cnn(x)
        output = torch.flatten(output, 1)
        feat = self.predicate_fc(predicate)
        combined = torch.cat((output, feat), dim = 1)
        output = self.fc(combined)
        
        return output
    
def init_weights(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

''' Step 3: Train the model. '''

INPUT_SIZE = 224 * 224 * 3  # Flattened size of input images
NUM_CLASSES = len(animal_dataset.classes)
NUM_PREDICATES = predicate_matrix.shape[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-4
WEIGHT = None # Tensor of size NUM_CLASSES and floating point dtype
REDUCTION = 'mean' # 'none' | 'mean' | 'sum'
LABEL_SMOOTHING = 0.0
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 20
H = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": []
}

print("[INFO]: Initialising the AnimalClassifier model...")

# model
model = AnimalClassifier(INPUT_SIZE, NUM_CLASSES, NUM_PREDICATES).to(DEVICE)
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

    i = 0

    for images, labels in train_loader:

        print("Training:", epoch + 1, i)

        # send input to "device"
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        predicates = predicate_matrix[labels].to(DEVICE)

        outputs = model(images, predicates)

        loss = lossfn(outputs, labels)

        optimizer.zero_grad() # setting all gradients to zero
        loss.backward() # computing gradients 
        optimizer.step() # updating weights

        total_train_loss += loss.item()
        train_correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()

        i += 1

    train_loss = total_train_loss/len(train_loader.dataset)
    train_accuracy = train_correct/len(train_loader.dataset)

    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] - Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")
    
    # evaluating with validation dataset

    with torch.no_grad():

        # set model in "eval" mode
        model.eval()

        i = 0

        for images, labels in val_loader:

            print("Validation:", epoch + 1, i)

            images, labels = images.to(DEVICE), labels.to(DEVICE)

            predicates = predicate_matrix[labels]

            outputs = model(images, predicates)

            loss = lossfn(outputs, labels)

            total_val_loss += loss.item()
            val_correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()

            i += 1

        val_loss = total_val_loss/len(val_loader.dataset)
        val_accuracy = val_correct/len(val_loader.dataset)

        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] - Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    H['train_loss'].append(train_loss)
    H['train_accuracy'].append(train_accuracy)
    H['val_loss'].append(val_loss)
    H['val_accuracy'].append(val_accuracy)

    scheduler.step()

end_time = datetime.now()
time_taken = end_time - start_time

print(f"[INFO]: Completed training in {time_taken.seconds} seconds and {time_taken.microseconds} microseconds!")

''' Step 4: Visualize the loss and accuracy on a graph. '''

print("[INFO]: Visualising the plot...")

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

print("[INFO]: Testing the model...")

with torch.no_grad():

    # set model in evaluation mode
    model.eval()

    test_images = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
    test_predictions = []

    i = 0

    for img_name in test_images:

        print("Test:", i)
        
        img_path = os.path.join(test_dir, img_name)
        
        image = Image.open(img_path).convert('RGB')
        
        image = transform(image).unsqueeze(0).to(DEVICE)

        dummy_predicate = -torch.ones(1, NUM_PREDICATES).to(DEVICE)

        outputs = model(image, dummy_predicate)
        predicted_class = torch.argmax(outputs, dim = 1).item()
        test_predictions.append((img_name, animal_dataset.classes[predicted_class]))

        i += 1


'''Step 6: Save predictions to a CSV file'''

print("[INFO]: Saving predictions to predicitons.csv...")

submission = pd.DataFrame(test_predictions, columns = ['image_id', 'class'])
submission.to_csv("predictions.csv", index = False)

print("[INFO]: Program completed.")