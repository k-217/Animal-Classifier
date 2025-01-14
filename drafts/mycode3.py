# testing for unseen classes

# evaluation metrics (scikit-learn)

# epoch logging (tqdm)

# check parameters and hyperparameters used (and anything default that could be modified for better)

# load test data in batches? and, applying the same transforms as training?

# class for dataset and dataloader

# pre-trained model

# documentation

import os
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime

from torchvision import datasets, transforms, models

import torch

from PIL import Image

DEVICE = torch.device("xpu" if torch.xpu.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

''' Step 1: Define the path to the training and test directories '''

train_dir = "train"
test_dir = "test"
predicate_matrix_dir = "predicate-matrix-binary.txt"
classes_dir = "classes.txt"

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(),
    transforms.RandomAffine(degrees = 15, scale = (0.9, 1.1), translate = (0.1, 0.1)),
    transforms.GaussianBlur(kernel_size = (3, 3)),
    transforms.Resize((224, 224), interpolation = transforms.InterpolationMode.BILINEAR, max_size = None, antialias = True),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

BATCH_SIZE = 32

print("[INFO]: Loading animal images dataset...")

animal_dataset = datasets.ImageFolder(root = train_dir, transform = transform) 

train_size = int(0.8 * len(animal_dataset))
val_size = len(animal_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(animal_dataset, [train_size, val_size], generator = torch.Generator().manual_seed(42))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = False) 

trainSteps = len(train_loader.dataset) // BATCH_SIZE
valSteps = len(val_loader.dataset) // BATCH_SIZE

predicate_matrix = []

with open(predicate_matrix_dir) as f:
    lines = f.readlines()

for line in lines:
    values = line.strip().split()
    values = [float(i) for i in values]
    predicate_matrix.append(values)

predicate_matrix = torch.tensor(predicate_matrix, dtype = torch.float32).to(DEVICE)

''' Step 2: Define a model architecture for image classification. '''

class AnimalClassifier(torch.nn.Module):
    
    def __init__(self, input_size, num_classes, num_predicates) -> None:

        super(AnimalClassifier, self).__init__()

        self.cnn = torch.nn.Sequential(

            # 5 sets of CONV => RELU => POOL layers

            # 3 RGB channels, 8 filters, 3x3 kernel, padding to keep size

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

        self._dummy_input = torch.zeros(1, 3, 224, 224)
        self._cnn_output_size = self._get_cnn_output_size(self._dummy_input)

        self.fc1 = torch.nn.Linear(in_features = self._cnn_output_size, out_features = 512)

        self.predicate_fc = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, num_predicates),
            torch.nn.Sigmoid()
        )

        self.classes_fc = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, num_classes) 
        )

    def forward(self, x):
        
        x = self.cnn(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        predicates = self.predicate_fc(x)
        classes = self.classes_fc(x)
        
        return predicates, classes
    
    def _get_cnn_output_size(self, x):
            with torch.no_grad():
                output = self.cnn(x)
                return output.size(1) * output.size(2) * output.size(3)

    
def init_weights(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

''' Step 3: Train the model. '''

INPUT_SIZE = 224 * 224 * 3
NUM_CLASSES = len(animal_dataset.classes)
NUM_PREDICATES = predicate_matrix.shape[1]
LEARNING_RATE = 1e-2
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

# loss functions: 
# CrossEntropyLoss = Negative Log Likelihood Loss + Log Softmax
# BCELoss = Binary Cross Entropy Loss
lossfn_classes = torch.nn.CrossEntropyLoss()
lossfn_predicates = torch.nn.BCELoss() 

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)

# scheduler to update learning rate after each epoch
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

print("[INFO]: Training the network...")

start_time = datetime.now()

# learning algorithm: backpropagation

best_val_accuracy = 0

for epoch in range(NUM_EPOCHS):

    # set model in "train" mode
    model.train()

    total_train_loss = 0
    train_correct = 0

    total_val_loss = 0
    val_correct = 0

    i = 1

    for images, labels in train_loader:

        print(f"[INFO]: Training - {epoch + 1}.{i}")

        # send input to "device"
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        predicates_ = predicate_matrix[labels].to(DEVICE)

        pred_predicates, pred_classes = model(images)

        loss_predicates = lossfn_predicates(pred_predicates, predicates_)
        loss_classes = lossfn_classes(pred_classes, labels)
        loss = 0.5 * loss_predicates + 0.5 * loss_classes

        optimizer.zero_grad() # setting all gradients to zero
        loss.backward() # computing gradients 
        optimizer.step() # updating weights

        total_train_loss += loss.item()
        train_correct += (pred_classes.argmax(1) == labels).type(torch.float).sum().item()

        i += 1

    train_loss = total_train_loss/len(train_loader.dataset)
    train_accuracy = train_correct/len(train_loader.dataset)

    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] - Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")
    
    # evaluating with validation dataset

    with torch.no_grad():

        # set model in "eval" mode
        model.eval()

        i = 1

        for images, labels in val_loader:

            print(f"[INFO]: Validation - {epoch + 1}.{i}")

            images, labels = images.to(DEVICE), labels.to(DEVICE)

            predicates_ = predicate_matrix[labels].to(DEVICE)

            pred_predicates, pred_classes = model(images)

            loss_predicates = lossfn_predicates(pred_predicates, predicates_)
            loss_classes = lossfn_classes(pred_classes, labels)
            loss = 0.5 * loss_predicates + 0.5 * loss_classes

            total_val_loss += loss.item()
            val_correct += (pred_classes.argmax(1) == labels).type(torch.float).sum().item()

            i += 1

        val_loss = total_val_loss/len(val_loader.dataset)
        val_accuracy = val_correct/len(val_loader.dataset)

        # if val_accuracy > best_val_accuracy:
        #     best_val_accuracy = val_accuracy
        #     states = {
        #         'model': model.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #         'scheduler': scheduler.state_dict(),
        #         'epoch': epoch,
        #         'best_val_accuracy': best_val_accuracy
        #     }
        #     torch.save(states, 'best_model.pth')

        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] - Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    H['train_loss'].append(train_loss)
    H['train_accuracy'].append(train_accuracy)
    H['val_loss'].append(val_loss)
    H['val_accuracy'].append(val_accuracy)

    scheduler.step(val_loss)

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

test_images = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
test_predictions = []

with torch.no_grad():

    # set model in evaluation mode
    model.eval()

    # can send images in batches

    for img_name in test_images:

        # zero-shot learning: leveraging semantic relationships or attribute-based representations (like predicates) to infer the labels of unseen classes.

        print(f"[INFO]: Testing image {img_name}...")
        
        img_path = os.path.join(test_dir, img_name)
        
        image = Image.open(img_path).convert('RGB')
        
        image = transform(image).unsqueeze(0).to(DEVICE)

        pred_predicates, pred_classes = model(image)

        predicted_class = torch.argmax(pred_classes).item()

        test_predictions.append((img_name, animal_dataset.classes[predicted_class]))


''' Step 6: Save predictions to a CSV file. '''

print("[INFO]: Saving predictions to predictions.csv...")

submission = pd.DataFrame(test_predictions, columns = ['image_id', 'class'])
submission.to_csv("predictions.csv", index = False)

print("[INFO]: Program completed.")