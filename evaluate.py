import csv

import tensorflow as tf
import numpy as np

import dataset_utils
import dnn_models
import params

dnn_models.config_gpu(1)

params.BATCH_SIZE = 1

run = 'logdir_12.03.2020..17.17'
model_dir = '/home/alonlahav/git-projects/dl_course_iWildCam/' + run

model = dnn_models.get_model()
model.load_weights(model_dir + '/model/mymodel_1/variables/variables')

if 0:   # Validate
  dataset = dataset_utils.get_dataset(locations=params.test_locations ,train=False)
  csv_name = 'val'
elif 1: # Train
  dataset = dataset_utils.get_dataset(locations=params.train_locations ,train=False)
  csv_name = 'train'
else:   # Test
  dataset = dataset_utils.get_dataset(locations='kaggle-test' ,train=False)
  csv_name = 'test'


results = []
m = tf.keras.metrics.CategoricalAccuracy()
i = 0
for im, lb in dataset:
  pred = model.predict(im)
  _ = m.update_state(lb, pred)
  print(m.result().numpy())
  category = np.argmax(pred)
  id = '?'
  results.append([i, id, category])
  i += 1
  #if i > 50:
  #  break

with open(model_dir + '/' + csv_name + '.csv', 'w', newline='') as fd:
  for r in results:
    fd.write(str(r[0]) + ' , ' + r[1] + ' , ' + str(r[2]) + '\n')

