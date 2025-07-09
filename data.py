import torch
import pandas as pd

# Load the data
print("Loading data...")
data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Create the dataset
print("Creating dataset...")
X = data.drop(columns=['label'])
y = data['label']
X = torch.tensor(X.values, dtype=torch.int32)
y = torch.tensor(y.values, dtype=torch.long)
X = X / 255 # Normalize the features to [0, 1] range
train_dataset = torch.utils.data.TensorDataset(X, y)

X_test = test_data
X_test = torch.tensor(X_test.values, dtype=torch.int32)
X_test = X_test / 255 # Normalize the features to [0, 1] range

# Save the datasets
print("Saving datasets to file...")
torch.save(train_dataset, 'train_dataset.pt')
torch.save(X_test, 'test_dataset.pt')