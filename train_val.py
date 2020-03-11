import time, datetime

import seaborn as sns
import numpy as np
import pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
from tensorflow import keras

import dataset_utils
import params

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


if params.VERY_SMALL_DATASET:
  locations = np.arange(10)
  params.SPLIT_TRAIN_TH = 5
else:
  locations = np.arange(139)
locations = np.random.permutation(locations)
train_dataset = dataset_utils.get_dataset(locations=locations[:params.SPLIT_TRAIN_TH])
test_dataset = dataset_utils.get_dataset(locations=locations[params.SPLIT_TRAIN_TH:], train=False)

if params.EXPLORE_DATASET:
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

if params.SPLIT[0] != 1:
  inputs = keras.Input(shape=(params.IMAGE_SHAPE[0] * params.SPLIT[0], params.IMAGE_SHAPE[1] * params.SPLIT[1], 3))
  x1 = feature_extractor_layer(inputs[:, :params.IMAGE_SHAPE[0],                        :params.IMAGE_SHAPE[1]]                )
  x2 = feature_extractor_layer(inputs[:, params.IMAGE_SHAPE[0]:params.IMAGE_SHAPE[0]*2, :params.IMAGE_SHAPE[1]]                )
  x3 = feature_extractor_layer(inputs[:, :params.IMAGE_SHAPE[0],                        params.IMAGE_SHAPE[1]:params.IMAGE_SHAPE[1]*2])
  x4 = feature_extractor_layer(inputs[:, params.IMAGE_SHAPE[0]:params.IMAGE_SHAPE[0]*2, params.IMAGE_SHAPE[1]:params.IMAGE_SHAPE[1]*2])
  pooling = (x1 + x2 + x3 + x4) / 4
  outputs = layers.Dense(params.num_classes)(pooling)
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
history = model.fit(train_dataset, epochs=params.N_EPOCHS,
                    callbacks=callbacks,
                    validation_data=test_dataset)
print('Training time:', time.time() - tb)

all_confusion = np.zeros((params.num_classes, params.num_classes), dtype=np.int)
for im, label, location in test_dataset:
  y_pred = tf.argmax(model.predict(im), axis=1)
  l = tf.argmax(label, axis=1)
  con_mat = tf.math.confusion_matrix(labels=l, predictions=y_pred, num_classes=params.num_classes).numpy()
  all_confusion += con_mat
all_confusion = all_confusion / all_confusion.sum(axis=1)[:, np.newaxis]
print(all_confusion)

figure = plt.figure()
sns.heatmap(all_confusion, annot=True,cmap=plt.cm.Blues)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')

plt.show()



