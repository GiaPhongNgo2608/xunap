import torch

BATCH_SIZE = 8 # increase / decrease according to GPU memeory
RESIZE_TO = 1024 # resize the image for training and transforms, default: 416
NUM_EPOCHS = 5 # number of epochs to train for
NUM_WORKERS = 2

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 

# training images and XML files directory
TRAIN_DIR = 'data/dataset/train'

# validation images and XML files directory
VALID_DIR = 'data/dataset/valid'

# classes: 0 index is reserved for background
CLASSES = [
    '__background__', 'A', 'B', 'C'
]

NUM_CLASSES = len(CLASSES)

# whether to visualize images after crearing the data loaders
VISUALIZE_TRANSFORMED_IMAGES = True

# location to save model and plots
OUT_DIR = 'outputs'