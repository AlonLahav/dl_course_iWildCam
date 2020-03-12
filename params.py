import numpy as np

np.random.seed(0)

MODEL2USE = 'SPATIAL' # SPLIT_2x2 / VANILA / SPATIAL
feature_extractor_url = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4'
#feature_extractor_url = 'https://tfhub.dev/google/imagenet/resnet_v1_101/feature_vector/4'

EXPLORE_DATASET = 0
SHOW_IMAGES = 0
IMAGE_SHAPE = (512, 1024)
IMAGE_SHAPE = (256, 512)
BATCH_SIZE = 4
SMALL_DATASET = 0
VERY_SMALL_DATASET = 0

N_EPOCHS = 500
SPLIT_TRAIN_TH = 100

SPLIT = (2, 2)

num_classes = 23

if VERY_SMALL_DATASET:
  locations = np.arange(10)
  SPLIT_TRAIN_TH = 5
else:
  locations = np.arange(139)
locations = np.random.permutation(locations)

train_locations = locations[:SPLIT_TRAIN_TH]
test_locations = locations[SPLIT_TRAIN_TH:]

