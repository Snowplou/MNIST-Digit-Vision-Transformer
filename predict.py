import torch
from torch import nn
import pandas as pd
from tqdm import tqdm
from config import *
from dataset import *
from model import *

# Check if a GPU is available and use it if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the data
print("Loading data...")
X = pd.read_csv('test.csv')

# Create the dataset
print("Creating dataset...")
X = torch.tensor(X.values, dtype=torch.int32)
X = X / 255 # Normalize the features to [0, 1] range
X = (X - mnist_mean[0]) / mnist_std[0]  # Normalize using the mean and std from the config

# Load the model
print("Loading model...")
model = MNISTModel()
model.to(device)
model.load_state_dict(torch.load('digit_model.pth', map_location=device))
model.eval() # Set the model to evaluation mode

# Predict the labels for the images
print("Predicting labels...")
predicted_labels = []
with torch.no_grad():
    X = X.to(device)
    batch_size = 64
    for i in tqdm(range(0, X.size(0), batch_size)):
        batch = X[i:i + batch_size]
        outputs = model(batch)
        _, predicted = torch.max(outputs, 1)
        predicted_labels.extend(predicted.cpu().numpy())

# Save the predictions to a CSV file with format ImageId,Label
print("Saving predictions to file...")
submission_df = pd.DataFrame({
    'ImageId': range(1, len(predicted_labels) + 1),
    'Label': predicted_labels
})
submission_df.to_csv('submission.csv', index=False)