from torch import nn, zeros, flatten, no_grad
from torch import device, xpu, cuda
import torchvision.models as models

'''

This is a basic module definition that inherits the base class for the Pytorch neural network modules.

It has a multi-task learning approach to the classification problem that learns both
    - image features (Pytorch tensor of its pixel values): single-label
    - predicate features (semantics with the picture): multi-label

Instead, it may also be termed as auxiliary task learning where predicate features are used to improve the performance of the model.

Args:
    - num_classes (int): number of categories of classification 
    - num_predicates (int): number of known semantic features of the classes


Architecture:

    Five convolutional layers: Conv => Batch Normalisation => ReLU activation => max pooling (applied to the image features)

    Fully connected layers for classification and predicate prediction.

Considering the standard dimension 224x224x3 of images features (as used by several CNN models), 
the tensor dimensions after each step is added along it.

'''

DEVICE = device("xpu" if xpu.is_available() else "cuda" if cuda.is_available() else "cpu")

class AnimalClassifier(nn.Module):
        
        def __init__(self, num_classes, num_predicates) -> None:

            super(AnimalClassifier, self).__init__()

            self.cnn = nn.Sequential(

                # initial: 224x224x3

                nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, padding = 1), # 224x224x16
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = (2, 2), stride = 2), # 112x112x16

                nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 1), # 112x112x32
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = (2, 2), stride = 2), # 56x56x32

                nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1), # 56x56x64
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = (2, 2), stride = 2), # 28x28x64

                nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1), # 28x28x128
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = (2, 2), stride = 2), # 14x14x128

                nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, padding = 1), # 14x14x256
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = (2, 2), stride = 2) # 7x7x256
            )

            self._dummy_input = zeros(1, 3, 224, 224).to(DEVICE)
            self._cnn_output_size = self._get_cnn_output_size(self._dummy_input)

            self.fc1 = nn.Linear(in_features = self._cnn_output_size, out_features = 512)

            self.predicate_fc = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, num_predicates),
                nn.Sigmoid() # Ensures outputs are between 0 and 1 for multi-label tasks
            )

            self.classes_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, num_classes) 
            )

        def forward(self, x): # Forward pass through the network

            '''
            
            Args:
                x (Tensor): input image features (batch_size, 3, 224, 224)

            Returns:
                tuple: predictions for predicates (multi-label) and classes (single-label)

            '''
            
            x = self.cnn(x)
            x = flatten(x, 1)
            x = self.fc1(x)
            predicates = self.predicate_fc(x)
            classes = self.classes_fc(x)
            
            return predicates, classes
        
        def _get_cnn_output_size(self, x): # to calculate the size of image tensor after convolution
                with no_grad():
                    output = self.cnn(x)
                    return output.size(1) * output.size(2) * output.size(3)