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

log_dir = './logdir__random_split' + datetime.datetime.now().strftime("%d.%m.%Y..%H.%M")
callbacks = [
  keras.callbacks.TensorBoard(log_dir=log_dir),
  batch_stats_callback,
  keras.callbacks.ModelCheckpoint(
      filepath=log_dir + '/model/mymodel_{epoch}',
      save_best_only=False,
      monitor='val_loss',
      verbose=1)
]

# Datasets definition
# -------------------
train_dataset, trn_size = dataset_utils.get_dataset(train=True)
test_dataset, tst_size = dataset_utils.get_dataset(train=False)
print('Train size:', trn_size)
print('Validation size:', tst_size)

# Get , show and prepare the model
# --------------------------------
model = dnn_models.get_model()
tf.keras.utils.plot_model(model, to_file='model.png')

if 0: # Some tests
  im, lbl = iter(train_dataset).next()
  f1 = model(im)
  loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
  l = loss(lbl, f1)

model.summary()

model.compile(
  optimizer=tf.keras.optimizers.Adam(),
  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
  metrics=['acc'])

# Training
# --------
tb = time.time()
history = model.fit(train_dataset, epochs=params.N_EPOCHS,
                    callbacks=callbacks,
                    validation_data=test_dataset)
print('Training time:', time.time() - tb)



