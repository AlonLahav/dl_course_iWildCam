import numpy as np

np.random.seed(0)

MODEL2USE = 'SPATIAL' # SPLIT_2x2 / VANILA / SPATIAL
feature_extractor_url = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4'
#feature_extractor_url = 'https://tfhub.dev/google/imagenet/resnet_v1_101/feature_vector/4'

IMAGE_SHAPE = (512, 1024) # Resizes the image to this shape before entering it to the DNN
BATCH_SIZE = 4

N_EPOCHS = 500

num_classes = 23


