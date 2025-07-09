import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from config import *
from dataset import AugmentedTensorDataset
from model import *

def main(args):
    # --- AUGMENTATION HYPERPARAMETERS ---
    # Use the mean and std for the MNIST dataset. For grayscale, they are single values.
    train_transforms = transforms.Compose([
        # This expects a PIL Image, so we'll convert our tensor to one first.
        transforms.ToPILImage(),
        transforms.RandomAffine(
            degrees=25,
            translate=(0.35, 0.35),
            scale=(0.6, 1.4),
            shear=30
        ),
        transforms.ToTensor(), # Converts back to a tensor and scales to [0.0, 1.0]
        transforms.Normalize(mean=mnist_mean, std=mnist_std)
    ])

    # Set the seed to 0
    torch.manual_seed(0)

    # Load the train dataset
    print("Loading dataset...")
    raw_train_dataset = torch.load("train_dataset.pt", weights_only=False)
    augmented_train_dataset = AugmentedTensorDataset(raw_train_dataset, transform=train_transforms)
    train_loader = DataLoader(augmented_train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Check if a GPU is available and use it if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create the model
    net = MNISTModel()

    # Load existing model weights if the continue flag is set
    if args.continue_training:
        try:
            net.load_state_dict(torch.load('digit_model.pth', map_location=device))
            print("Continue flag set. Loaded existing model weights to continue training.")
        except FileNotFoundError:
            print("Continue flag was set, but 'digit_model.pth' not found. Starting from scratch.")
    else:
        print("Starting new training from scratch.")

    # Move the model to the device
    net.to(device)

    # Print the total number of parameters
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Total parameters: {total_params:,}")

    # Get the input and output dimensions of the current model
    testing_input = augmented_train_dataset[0][0].unsqueeze(0).to(device) # Get a single sample and move it to the device
    print("Input shape: ", testing_input.shape)
    output = net(testing_input)
    print("Output shape: ", output.shape)

    # Train the model
    print("Training model...")
    net.train() # Set the model to training mode
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    num_epochs = 10
    for epoch in range(num_epochs):
        bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}", dynamic_ncols=True)
        for i, (X_batch, y_batch) in bar:
        # for i, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = net(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            bar.set_postfix(loss=f"{loss.item():.4f}")
        # print(f"Epoch {epoch+1}, loss: {loss.item():.4f}")

    # Save the trained model
    print("Saving the trained model...")
    torch.save(net.state_dict(), 'digit_model.pth')



if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Train a MNIST digit recognition model.")

    # Add the --continue argument
    # action='store_true' means if the flag is present, the value is True. Otherwise, it's False.
    parser.add_argument(
        "-c", "--continue_training",
        action="store_true",
        help="Continue training from a saved model (digit_model.pth)"
    )

    # Parse the arguments from the command line
    args = parser.parse_args()

    # Run the main function with the parsed arguments
    main(args)