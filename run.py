import torch
from torch import nn
import pygame
import pandas as pd
from config import *
from dataset import *
from model import *

index_to_display = -1 # Index of the image to display

# Use a cpu to make it easier to run since run.py doesn't need to be fast
device ="cpu"
print(f"Using device: {device}")

# Load the data
print("Loading data...")
data = pd.read_csv('train.csv')

# Create the dataset
print("Creating dataset...")
X = data.drop(columns=['label'])
y = data['label']
X = torch.tensor(X.values, dtype=torch.int32)
y = torch.tensor(y.values, dtype=torch.long)
X = X / 255 # Normalize the features to [0, 1] range
original_image = X[index_to_display].reshape(28, 28)

# Normalize the data using the mean and std from the config
X = (X - mnist_mean[0]) / mnist_std[0]

normalized_image = X[index_to_display].reshape(28, 28)
label = y[index_to_display] # The digit the image represents

# Load the model
print("Loading model...")
model = MNISTModel()
model.load_state_dict(torch.load('digit_model.pth', map_location=device))
model.eval() # Set the model to evaluation mode

# Predict the label for the image
print("Predicting label...")
with torch.no_grad():
    output = model(normalized_image)[0]
    print("Raw output:", output)
    softmax_output = torch.softmax(output, dim=0)
    torch.set_printoptions(sci_mode=False) # No scientific notation
    print("Softmax output:", softmax_output)
    predicted_label = torch.argmax(softmax_output).item()
    print("Predicted label:", predicted_label)
    print("Actual label:", label.item())

# Display the image using pygame
print("Displaying image...")
pygame.init()
screen = pygame.display.set_mode((280, 280))
pygame.display.set_caption('MNIST Image')
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))  # Fill the screen with white
    for i in range(28):
        for j in range(28):
            color = (original_image[i, j].item() * 255, original_image[i, j].item() * 255, original_image[i, j].item() * 255)
            pygame.draw.rect(screen, color, (j * 10, i * 10, 10, 10))

    pygame.display.flip()
    pygame.time.delay(100)

pygame.quit()