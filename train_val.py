import time

import numpy as np
import pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers

import dataset_exploration

EXPORT_MODEL = False
EXPLORE_DATASET = False
SHOW_IMAGES = False

IMAGE_SHAPE = (224, 224)
num_classes = 23
np.random.seed(0)
tf.random.set_seed(0)

def config_gpu(use_gpu=True):
  try:
    if use_gpu:
      gpus = tf.config.experimental.list_physical_devices('GPU')
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    else:
      os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
  except:
    pass

# Define class to collect some training data
class CollectBatchStats(tf.keras.callbacks.Callback):
  def __init__(self):
    self.batch_losses = []
    self.batch_acc = []

  def on_train_batch_end(self, batch, logs=None):
    if 'loss' in logs.keys():
      self.batch_losses.append(logs['loss'])
      self.batch_acc.append(logs['acc'])
    self.model.reset_metrics()

config_gpu(True)

# Get the data
# ------------
root_path_ds = '/media/alonlahav/4T-b/datasets/iwildcam-2019-fgvc6'
def process_path(label, file_name, location, n_frames):
  if EXPLORE_DATASET and not SHOW_IMAGES:
    image = np.zeros(IMAGE_SHAPE)
  else:
    fn = root_path_ds + '/train_images/' + file_name
    image = tf.io.read_file(fn)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, IMAGE_SHAPE)
  label = tf.one_hot(label, num_classes)

  if 0:
    image = tf.cond(tf.math.equal(n_frames, 1), lambda: tf.image.grayscale_to_rgb(image),
                    lambda: tf.identity(image))
  if 0:
    image = tf.cond(tf.math.equal(n_frames, 1), lambda: tf.concat((image, image, image), axis=2),
                    lambda: tf.identity(image))

  return image, label, location

train_csv_file = root_path_ds + '/train.csv'
record_defaults = [-1, '--', -1, -1]
dataset = tf.data.experimental.CsvDataset(train_csv_file, record_defaults, select_cols=[0, 2, 5, 8], header=True) # category-id / file name / location
if not EXPLORE_DATASET:
  #dataset = dataset.filter(lambda image, label, location, n_frames: (n_frames == 3) and (location == 32))
  dataset = dataset.filter(lambda image, label, location, n_frames: (n_frames == 3))
dataset = dataset.map(process_path)
if EXPLORE_DATASET:
  dataset_exploration.explore_dataset(dataset, num_classes, SHOW_IMAGES)
dataset = dataset.batch(32)

if 0: # Go over the dataset
  for images, labels, locations in dataset:
    for image, label, location in zip(images, labels, locations):
      #print(image.shape)
      assert image.shape[2] == 3
      #continue
      #if location != 33:
      #  continue
      print(image.shape, label.numpy())
      plt.imshow(image)
      plt.title(label.numpy())
      plt.waitforbuttonpress()

# Get features extractor and define the model
# -------------------------------------------
feature_extractor_url = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4'
feature_extractor_layer = hub.KerasLayer(feature_extractor_url,
                                         input_shape=(224,224,3))

#feature_batch = feature_extractor_layer(image_batch)
feature_extractor_layer.trainable = False

model = tf.keras.Sequential([
  feature_extractor_layer,
  layers.Dense(num_classes)
])

model.summary()

model.compile(
  optimizer=tf.keras.optimizers.Adam(),
  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
  metrics=['acc'])

batch_stats_callback = CollectBatchStats()

history = model.fit(dataset, epochs=2,
                    callbacks = [batch_stats_callback])

plt.figure()
plt.ylabel("Loss")
plt.xlabel("Training Steps")
#plt.ylim([0,2])
plt.plot(batch_stats_callback.batch_losses)

plt.figure()
plt.ylabel("Accuracy")
plt.xlabel("Training Steps")
plt.ylim([0,1])
plt.plot(batch_stats_callback.batch_acc)

plt.show()

# Check prediction
# ----------------


# Export model
# ------------
if EXPORT_MODEL:
  export_path = "/tmp/saved_models/{}".format(int(time.time()))
  model.save(export_path, save_format='tf')
  print(export_path)

# Check exported model


