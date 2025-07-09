# INPUTS
mnist_mean = (0.1307,)
mnist_std = (0.3081,)
INPUT_IMAGE_SIZE = (28, 28) # MNIST images are 28x28 pixels

# HYPERPARAMETERS
PATCH_SIZE = (4, 4) # Size of the patches to be extracted from the images
IMAGE_SIZE = ((INPUT_IMAGE_SIZE[0] // PATCH_SIZE[0]), (INPUT_IMAGE_SIZE[1] // PATCH_SIZE[1])) # Image size after patching (patch x patch instead of pixel x pixel)
NUM_CLASSES = 10 # Predicts digits 0-9, so 10 possible classes
EMBED_DIM = 128 # Dimension of the embedding layer
NUM_ATTENTION_LAYERS = 8 # Number of attention layers in the model
NUM_HEADS = 4 # Number of attention heads in each block

BATCH_SIZE = 512 # Batch size for training
LEARNING_RATE = 0.0001 # Learning rate for the optimizer

# VALIDATION
if INPUT_IMAGE_SIZE[0] % PATCH_SIZE[0] != 0 or INPUT_IMAGE_SIZE[1] % PATCH_SIZE[1] != 0:
    raise ValueError("INPUT_IMAGE_SIZE must be divisible by PATCH_SIZE for the model to work properly.")
if EMBED_DIM % NUM_HEADS != 0:
    raise ValueError("EMBED_DIM must be divisible by NUM_HEADS for the model to work properly.")