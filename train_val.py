import time, datetime

import seaborn as sns
import numpy as np
import pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
from tensorflow import keras

import dataset_exploration

N_EPOCHS = 500
EXPLORE_DATASET = 0
SHOW_IMAGES = 0
SPLIT_TRAIN_TH = 100
SMALL_DATASET = 0
VERY_SMALL_DATASET = 0

IMAGE_SHAPE = (224, 224)
SPLIT = (2, 2)
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

batch_stats_callback = CollectBatchStats()

log_dir = './' + datetime.datetime.now().strftime("%d.%m.%Y..%H.%M")
callbacks = [
  keras.callbacks.TensorBoard(log_dir=log_dir),
  batch_stats_callback,
  keras.callbacks.ModelCheckpoint(
      filepath=log_dir + '/model/mymodel_{epoch}',
      save_best_only=True,
      monitor='val_loss',
      verbose=1)
]


config_gpu(True)

# Get the data
# ------------
root_path_ds = '/media/alonlahav/4T-b/datasets/iwildcam-2019-fgvc6'

def process_path_no_augmentation(label, file_name, location, n_frames):
  fn = root_path_ds + '/train_images/' + file_name
  image = tf.io.read_file(fn)
  image = tf.image.decode_jpeg(image)
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = tf.image.resize(image, (IMAGE_SHAPE[0] * SPLIT[0], IMAGE_SHAPE[0] * SPLIT[1]))
  label = tf.one_hot(label, num_classes)

  return image, label, location

def process_path(label, file_name, location, n_frames):
  if EXPLORE_DATASET and not SHOW_IMAGES:
    image = np.zeros(IMAGE_SHAPE)
  else:
    fn = root_path_ds + '/train_images/' + file_name
    image = tf.io.read_file(fn)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    if not EXPLORE_DATASET:
      image = tf.image.resize(image, (IMAGE_SHAPE[0] * SPLIT[0], IMAGE_SHAPE[0] * SPLIT[1]))
  label = tf.one_hot(label, num_classes)

  if 0:
    image = tf.cond(tf.math.equal(n_frames, 1), lambda: tf.image.grayscale_to_rgb(image),
                    lambda: tf.identity(image))
  if 0:
    image = tf.cond(tf.math.equal(n_frames, 1), lambda: tf.concat((image, image, image), axis=2),
                    lambda: tf.identity(image))

  # data augmentation
  image = tf.image.random_flip_left_right(image)
  image = tf.image.random_hue(image, 0.5)
  image = tf.image.random_contrast(image, 0.25, 0.75)

  return image, label, location

def get_dataset(locations, train=True):
  def _filter_classes(image, label, location, n_frames):
    if label == 1:
      return True
    else:
      return False

  train_csv_file = root_path_ds + '/train.csv'
  record_defaults = [-1, '--', -1, -1]
  dataset = tf.data.experimental.CsvDataset(train_csv_file, record_defaults, select_cols=[0, 2, 5, 8], header=True) # category-id / file name / location
  dataset = dataset.filter(lambda label, image, location, n_frames: tf.reduce_any(location == locations))
  if not EXPLORE_DATASET:
    dataset = dataset.filter(lambda label, image, location, n_frames: (n_frames == 3))

  if SMALL_DATASET:
    dataset = dataset.filter(lambda label, image, location, n_frames: (label == 1 or label == 17 or label == 18))

  if train:
    dataset = dataset.map(process_path)
  else:
    dataset = dataset.map(process_path_no_augmentation)

  if EXPLORE_DATASET:
    dataset_exploration.explore_dataset(dataset, num_classes, SHOW_IMAGES)

  dataset = dataset.batch(32)

  return dataset

if VERY_SMALL_DATASET:
  locations = np.arange(10)
  SPLIT_TRAIN_TH = 5
else:
  locations = np.arange(139)
locations = np.random.permutation(locations)
train_dataset = get_dataset(locations=locations[:SPLIT_TRAIN_TH])
test_dataset = get_dataset(locations=locations[SPLIT_TRAIN_TH:], train=False)

if EXPLORE_DATASET:
  exit(0)

# Get features extractor and define the model
# -------------------------------------------
if 1:
  feature_extractor_url = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4'
if 0:
  feature_extractor_url = 'https://tfhub.dev/google/imagenet/resnet_v1_101/feature_vector/4'
feature_extractor_layer = hub.KerasLayer(feature_extractor_url,
                                         input_shape=(224,224,3))
feature_extractor_layer.trainable = False

if 0:
  im, lbl, loc = iter(train_dataset).next()
  feature_extractor_layer(im[:, :224, :224])

if SPLIT[0] != 1:
  inputs = keras.Input(shape=(IMAGE_SHAPE[0] * SPLIT[0], IMAGE_SHAPE[1] * SPLIT[1], 3))
  x1 = feature_extractor_layer(inputs[:, :IMAGE_SHAPE[0],                 :IMAGE_SHAPE[1]]                )
  x2 = feature_extractor_layer(inputs[:, IMAGE_SHAPE[0]:IMAGE_SHAPE[0]*2, :IMAGE_SHAPE[1]]                )
  x3 = feature_extractor_layer(inputs[:, :IMAGE_SHAPE[0],                 IMAGE_SHAPE[1]:IMAGE_SHAPE[1]*2])
  x4 = feature_extractor_layer(inputs[:, IMAGE_SHAPE[0]:IMAGE_SHAPE[0]*2, IMAGE_SHAPE[1]:IMAGE_SHAPE[1]*2])
  pooling = (x1 + x2 + x3 + x4) / 4
  outputs = layers.Dense(num_classes)(pooling)
  model = keras.Model(inputs=inputs, outputs=outputs, name='classification')
else:
  model = tf.keras.Sequential([
    feature_extractor_layer,
    layers.Dense(num_classes)
  ])

model.summary()

model.compile(
  optimizer=tf.keras.optimizers.Adam(),
  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
  metrics=['acc'])

tb = time.time()
history = model.fit(train_dataset, epochs=N_EPOCHS,
                    callbacks=callbacks,
                    validation_data=test_dataset)
print('Training time:', time.time() - tb)

all_confusion = np.zeros((num_classes, num_classes), dtype=np.int)
for im, label, location in test_dataset:
  y_pred = tf.argmax(model.predict(im), axis=1)
  l = tf.argmax(label, axis=1)
  con_mat = tf.math.confusion_matrix(labels=l, predictions=y_pred, num_classes=num_classes).numpy()
  all_confusion += con_mat
print(all_confusion)

figure = plt.figure()
sns.heatmap(all_confusion, annot=True,cmap=plt.cm.Blues)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')

plt.figure()
plt.ylabel("Loss")
plt.xlabel("Training Steps")
plt.plot(batch_stats_callback.batch_losses)

plt.figure()
plt.ylabel("Accuracy")
plt.xlabel("Training Steps")
plt.ylim([0,1])
plt.plot(batch_stats_callback.batch_acc)

plt.show()



