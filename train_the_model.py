'''

This file defines the hyperparameters requried during training and the training algorithm, i.e. Gradient Descent and Backpropagation. 
The best_model obtained after training the model is saved in best_model.pth.

Weight and Bias Initialisation: 

    Kaiming initialization ensures that the variance of activations is maintained across layers, 
    avoiding vanishing or exploding gradients during training 
    (an issue with the multiple ReLU activiation functions applied).

    Bias is set to zero.

Loss Function(s): 
    CrossEntropyLoss(): for the classes labels
    
    BinaryCrossEntropyLoss(): for the predicates predicted by the model
    
    lossfn_alignment() (Custom loss function): for the alignment of image features predicted and the predicates of the class. 
        - takes dot product of predicted class tensor from model with the predicate matrix, and
		- computes mse loss with this projection and the true predicates from the matrix

Optimizer:
    Adam: Advance Moment Estimation for gradient descent (optimal as a combination of SGD and RMSP)
    Learning Rate = 1e-3
    Weight Decay = 1e-4

Scheduler:
    OneCycleLR(): anneals the learning rate from an initial learning rate to some maximum learning rate and 
        then from that maximum learning rate to some minimum learning rate much lower than the initial learning rate.

Evaluation metrics:
    Loss: the weighted sum of the loss functions defined above
    Accuracy: the percentage of correct predictions

'''

import model_definition, loading_datasets

from torch import nn, optim

import torch

from datetime import datetime

from tqdm import tqdm

import matplotlib.pyplot as plt

def init_weights(m): # initializes weights and biases for the network
    
    ''' Args: m (model instance) '''

    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        
        nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu') 
        
        if m.bias is not None:

            # sets all bias to zero
            
            nn.init.zeros_(m.bias)


DEVICE = torch.device("xpu" if torch.xpu.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = len(loading_datasets.animal_classes)
NUM_PREDICATES = loading_datasets.predicate_matrix.shape[1]
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 60

H = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": []
}

print("[INFO]: Initialising the AnimalClassifier model...")

# model
model = model_definition.AnimalClassifier(NUM_CLASSES, NUM_PREDICATES).to(DEVICE)
model.apply(init_weights)

# loss functions: 
# CrossEntropyLoss = Negative Log Likelihood Loss + Log Softmax
# BCELoss = Binary Cross Entropy Loss

lossfn_classes = nn.CrossEntropyLoss() # for image features

lossfn_predicates = nn.BCELoss() # for predicate features

def normalize(tensor):
    return (tensor - tensor.mean(dim = 0)) / (tensor.std(dim = 0) + 1e-6)

def lossfn_alignment(pred_classes, predicate_matrix, labels):
    true_predicates = predicate_matrix[labels]
    pred_projection = normalize(torch.matmul(pred_classes, predicate_matrix))
    loss_alignment = torch.nn.functional.mse_loss(pred_projection, true_predicates)
    return loss_alignment/10

RATIO1 = 0.5 # Ratio of loss_predicates in loss
RATIO2 = 0.3 # Ratio of loss_classes in loss
RATIO3 = 0.2 # Ratio of loss_alignment in loss

# optimizer
optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)

# scheduler to update learning rate after each epoch
# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 5)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr = 0.01, steps_per_epoch = len(loading_datasets.train_loader), epochs = NUM_EPOCHS)

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

    train_bar = tqdm(loading_datasets.train_loader, desc = f"[INFO]: Epoch [{epoch + 1}/{NUM_EPOCHS}] - Training", leave = False)

    for i, (images, labels) in enumerate(train_bar):

        # send input to "device"
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        predicates_ = loading_datasets.predicate_matrix[labels].to(DEVICE)

        pred_predicates, pred_classes = model(images)

        loss_predicates = lossfn_predicates(pred_predicates, predicates_)
        loss_classes = lossfn_classes(pred_classes, labels)
        loss_alignment = lossfn_alignment(pred_classes, loading_datasets.predicate_matrix, labels)
        loss = RATIO1 * loss_predicates + RATIO2 * loss_classes + RATIO3 * loss_alignment

        optimizer.zero_grad() # setting all gradients to zero
        loss.backward() # computing gradients 
        optimizer.step() # updating weights

        total_train_loss += loss.item()
        train_correct += (pred_classes.argmax(1) == labels).type(torch.float).sum().item()

        train_bar.set_postfix(loss = f"{total_train_loss/len(loading_datasets.train_loader.dataset):.4f}", accuracy = train_correct/len(loading_datasets.train_loader.dataset), progress = f"{(i + 1) / len(loading_datasets.train_loader) * 100:.2f}%")

    train_loss = total_train_loss/len(loading_datasets.train_loader.dataset)
    train_accuracy = train_correct/len(loading_datasets.train_loader.dataset)

    print(f"[INFO]: Epoch [{epoch + 1}/{NUM_EPOCHS}] - Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")
    
    # evaluating with validation dataset

    with torch.no_grad():

        # set model in "eval" mode
        model.eval()

        val_bar = tqdm(loading_datasets.val_loader, desc = f"[INFO]: Epoch [{epoch + 1}/{NUM_EPOCHS}] - Validation", leave = False) # leave = False ensures the log disappears from the terminal after it completes

        for i, (images, labels) in enumerate(val_bar):

            images, labels = images.to(loading_datasets.DEVICE), labels.to(DEVICE)

            predicates_ = loading_datasets.predicate_matrix[labels].to(DEVICE)

            pred_predicates, pred_classes = model(images)

            loss_predicates = lossfn_predicates(pred_predicates, predicates_)
            loss_classes = lossfn_classes(pred_classes, labels)
            loss_alignment = lossfn_alignment(pred_classes, loading_datasets.predicate_matrix, labels)
            loss = RATIO1 * loss_predicates + RATIO2 * loss_classes + RATIO3 * loss_alignment

            total_val_loss += loss.item()
            val_correct += (pred_classes.argmax(1) == labels).type(torch.float).sum().item()

            val_bar.set_postfix(loss = f"{total_val_loss/len(loading_datasets.val_loader.dataset):.4f}", accuracy = val_correct/len(loading_datasets.val_loader.dataset), progress = f"{(i + 1) / len(loading_datasets.val_loader) * 100:.6f}%")

        val_loss = total_val_loss/len(loading_datasets.val_loader.dataset)
        val_accuracy = val_correct/len(loading_datasets.val_loader.dataset)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), "best_model.pth")

        print(f"[INFO]: Epoch [{epoch + 1}/{NUM_EPOCHS}] - Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    H['train_loss'].append(train_loss)
    H['train_accuracy'].append(train_accuracy)
    H['val_loss'].append(val_loss)
    H['val_accuracy'].append(val_accuracy)

    scheduler.step()

end_time = datetime.now()
time_taken = end_time - start_time

print(f"[INFO]: Completed training in {time_taken.seconds} seconds!")

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
