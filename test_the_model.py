'''

This module tests the model after training and saves predictions to predictions.csv.

The predictions are based upon the model outputs and handles the unseen classes in the following way:

Utilizing a simple zero-shot learning approach,

    calculate probability of a class by softmax on the classes returned by the model, max = predicted_class1

    take a dynamically weighted similarity: Intersection over Union and Cosine Similarity to check the predicates
    the class with higher similarity = predicted_class2

    if the predicted_class2 is in the unseen bunch, actual predicted class = predicted_class2

    if not,
        if the probability of that class is greater than a confidence threshold, actual predicted class = predicted_class2
	    if not, actual predicted class = predicted_class1

    return the actual predicted class

'''

import train_the_model, loading_datasets

import os

import pandas as pd

import torch

from PIL import Image

from tqdm import tqdm

print("[INFO]: Testing the model...")

confidence_threshold = 0.02
seen = 40 # seen classes

test_predictions = []

with torch.no_grad():

    # set model in evaluation mode
    train_the_model.model.eval()

    test_bar = tqdm(loading_datasets.test_images, desc = "[INFO]: Testing Images", leave   = True)

    for i, img_name in enumerate(test_bar):

        # zero-shot learning: leveraging semantic relationships or attribute-based representations (like predicates) to infer the labels of unseen classes.
        
        img_path = os.path.join(loading_datasets.test_dir, img_name)
        
        image = Image.open(img_path).convert('RGB')
        
        image = loading_datasets.transform(image).unsqueeze(0).to(loading_datasets.DEVICE)

        pred_predicates, pred_classes = train_the_model.model(image)

        # probability of a class

        class_probs = torch.nn.functional.softmax(pred_classes, dim = 1)
        max_class_prob, predicted_class1 = torch.max(class_probs, dim = 1)

        # computing similarity in order unseen classes may also be compared

        # Jaccard Similarity (Intersection over Union)
        intersection = torch.sum(pred_predicates * loading_datasets.predicate_matrix, dim = 1)
        union = torch.sum((pred_predicates + loading_datasets.predicate_matrix) > 0, dim = 1)
        similarity1 = (intersection + 1e-8) / (union + 1e-8)

        # Cosine Similarity (Dot product after normalization)
        similarity2 = torch.nn.functional.cosine_similarity(pred_predicates, loading_datasets.predicate_matrix.to(loading_datasets.DEVICE), dim = 1)

        similarity2 = (similarity2 + 1) / 2 # Normalisation

        # Dynamically adjust weights based on the difference in similarities
        # If IoU dominates (large IoU, small Cosine Similarity), increase IoU's weight
        # If Cosine dominates (small IoU, large Cosine Similarity), increase Cosine's weight
        diff = torch.abs(similarity1 - similarity2)
        ratio1 = torch.mean(1 - diff).item()  # IoU weight
        ratio2 = 1 - ratio1  # Cosine similarity weight

        similarity = ratio1 * similarity1 + ratio2 * similarity2

        max_similarity, predicted_class2 = torch.max(similarity, dim = 0)

        if predicted_class2.item() < seen:

            if class_probs[0][predicted_class2.item()] > confidence_threshold:
                predicted_class = predicted_class2.item()

            else:
                predicted_class = predicted_class1.item()
    
        else:

            predicted_class = predicted_class2.item()

        test_predictions.append((img_name, loading_datasets.animal_classes[predicted_class]))

        if (i + 1) % 10 == 0:
            test_bar.set_postfix(progress = f"{(i + 1) / len(loading_datasets.test_images) * 100:.2f}%")


print("[INFO]: Saving predictions to predictions.csv...")

submission = pd.DataFrame(test_predictions, columns = ['image_id', 'class'])
submission.to_csv("predictions.csv", index = False)

print("[INFO]: Program completed.")
