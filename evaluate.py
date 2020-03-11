
import tensorflow as tf

import dataset_utils
import dnn_models
import params

dnn_models.config_gpu(0)

run = 'logdir_11.03.2020..07.34'
model_dir = '/home/alonlahav/git-projects/dl_course_iWildCam/' + run

model = dnn_models.get_2x2_model()
model.load_weights(model_dir + '/model/mymodel_1/variables/variables')

if 0:   # Validate
  dataset = dataset_utils.get_dataset(locations=params.test_locations ,train=False)
elif 1: # Train
  dataset = dataset_utils.get_dataset(locations=params.train_locations ,train=False)
else:   # Test
  dataset = dataset_utils.get_dataset(locations='kaggle-test' ,train=False)


m = tf.keras.metrics.CategoricalAccuracy()
for im, lb in dataset:
  pred = model.predict(im)
  _ = m.update_state(lb, pred)
  print(m.result().numpy())
