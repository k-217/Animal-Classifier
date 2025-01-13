'''

Data Preprocessing: 
    - made changes to classes.txt and predicate-matrix-binary.txt
      to account for the missing classes in the training data
      and, hence the mismatched labels in the dataloader and the predicate matrix
    - split the data into training and validation set by torch.utils.data.random_split() with a 80-20 ratio
    - made batches of 32 images (random) to run mini-batch gradient descent in the training

Data Augmentation
    - applied transformations to the images
        - RandomHorizontalFlip(): randomly flips some of the images in the batches
        - ColorJitter(): randomly changes image brightness, contrast, saturation, and hue.
        - RandomAffine(): applies random affine transformations such as rotation, scaling, and translation
        - Resize(): resizes images to a standard size of 224x224 pixels using bilinear interpolation.
        - ToTensor(): converts PIL image into tensor with pixel values scaled to [0, 1]
        - Normalize(): normalizes the images using channel-wise mean and standard deviations, consistent with ImageNet standards

'''

# Importing necessary libraries

import os
from torchvision import datasets, transforms
from torch import Generator
from torch import tensor
from torch.utils.data import DataLoader, random_split 
import torch

# Defining the device for processing

DEVICE = torch.device("xpu" if torch.xpu.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# Defining the paths to different directories

base_dir = os.path.dirname(__file__)
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
predicate_matrix_dir = os.path.join(base_dir, 'predicate-matrix-binary.txt')
classes_dir = os.path.join(base_dir, 'classes.txt')

# list of transformations that will be applied to the images
# for data augmentation and generalization

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(),
    transforms.RandomAffine(degrees = 15, scale = (0.9, 1.1), translate = (0.1, 0.1)),
    transforms.Resize((224, 224), interpolation = transforms.InterpolationMode.BILINEAR, max_size = None, antialias = True),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]) # channel-wise means and sds used by several models
])

BATCH_SIZE = 32 # preparing for mini-batch gradient descent

print("[INFO]: Loading animal images dataset...")

animal_dataset = datasets.ImageFolder(root = train_dir, transform = transform) # stores and manages data 

split = 0.8
train_size = int(split * len(animal_dataset))
val_size = len(animal_dataset) - train_size
train_dataset, val_dataset = random_split(animal_dataset, [train_size, val_size], generator = Generator().manual_seed(42))

train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, pin_memory = DEVICE.type == 'cuda', num_workers = 4)
val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = False, pin_memory = DEVICE.type == 'cuda', num_workers = 4) 

test_images = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]

# storing predicate matrix to a pytorch tensor and sending it to device

print("[INFO]: Loading predicate-matrix-binary.txt...")

predicate_matrix = []

with open(predicate_matrix_dir) as f:
    lines = f.readlines()

for line in lines:
    values = line.strip().split()
    values = [float(i) for i in values]
    predicate_matrix.append(values)

predicate_matrix = tensor(predicate_matrix, dtype = torch.float32).to(DEVICE)

# storing string labels of animal classes in a python list

# animal_dataset.classes would have given a list quite similar,
# however, the absence of some of the classes in the training data
# makes it better to have all the classes (in the test data) this way. 

print("[INFO]: Loading classes.txt...")

animal_classes = []

with open(classes_dir) as f:
    lines = f.readlines()

for line in lines:
    animal_classes.append(line.strip().split()[-1])