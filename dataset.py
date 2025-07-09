from torch.utils.data import Dataset

class AugmentedTensorDataset(Dataset):
    def __init__(self, tensor_dataset, transform=None):
        self.tensor_dataset = tensor_dataset
        self.transform = transform

    def __getitem__(self, index):
        # Get the original image and label
        image, label = self.tensor_dataset[index]

        # Reshape to (channel, height, width)
        image = image.view(1, 28, 28)

        # Apply transformations if they exist
        if self.transform:
            image = self.transform(image)
        
        return image, label

    def __len__(self):
        return len(self.tensor_dataset)