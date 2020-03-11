import time, datetime

import seaborn as sns
import numpy as np
import pylab as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

import dataset_utils
import params
import dnn_models

np.random.seed(0)
tf.random.set_seed(0)

dnn_models.config_gpu()

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

log_dir = './logdir_' + datetime.datetime.now().strftime("%d.%m.%Y..%H.%M")
callbacks = [
  keras.callbacks.TensorBoard(log_dir=log_dir),
  batch_stats_callback,
  keras.callbacks.ModelCheckpoint(
      filepath=log_dir + '/model/mymodel_{epoch}',
      save_best_only=True,
      monitor='val_loss',
      verbose=1)
]


train_dataset = dataset_utils.get_dataset(locations=params.train_locations)
test_dataset = dataset_utils.get_dataset(locations=params.test_locations, train=False)

if params.EXPLORE_DATASET:
  exit(0)

# Get features extractor and define the model
# -------------------------------------------

if params.MODEL2USE == 'SPLIT_2x2':
  model = dnn_models.get_2x2_model()

if params.MODEL2USE == 'VANILA':
  model = dnn_models.get_vanila_model()

if 0:
  im, lbl, loc = iter(train_dataset).next()
  f0 = feature_extractor_layer(im[:, :224, :224])
  f1 = model(im)
  loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
  l = loss(lbl, f1)

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



