import csv
from tqdm import tqdm

import pylab as plt
import tensorflow as tf
import numpy as np
import seaborn as sns

import dataset_utils
import dnn_models
import params
from dataset_exploration import classe_names

np.random.seed(0)
tf.random.set_seed(10)

SHOW = 0

if 0: # Dump dummy CSV result file
  root_path_ds = '/media/alonlahav/4T-b/datasets/iwildcam-2019-fgvc6'
  test_csv_file = root_path_ds + '/test.csv'
  with open(test_csv_file, newline='') as csvfile:
    csv_in = csv.reader(csvfile, delimiter=',')
    with open('./dummy.csv', 'w', newline='') as fd:
      fd.write('Id,Predicted\n')
      i = 0
      for r in csv_in:
        if r[0] == 'date_captured':
          continue
        fd.write(r[3] + ',0' + '\n')
        i += 1
  exit(0)

dnn_models.config_gpu(1)

params.BATCH_SIZE = 4

run = 'logdir__random_split14.03.2020..15.56'
model_dir = '/home/alonlahav/git-projects/dl_course_iWildCam/' + run

model = dnn_models.get_model()
model.load_weights(model_dir + '/model/mymodel_49/variables/variables')

if 0:   # Validate
  dataset, db_size = dataset_utils.get_dataset(train=0)
  csv_name = 'val'
  max_items = max_items2u = 4000 # db_size
elif 0: # Train
  dataset, db_size = dataset_utils.get_dataset(train=1)
  csv_name = 'train'
  max_items = max_items2u = 4000 # db_size
else:   # Test, 153730 rows. This file should have a header row. CSV: integer-index , UUID, integer-result
  dataset, db_size = dataset_utils.get_dataset_kaggle_test()
  csv_name = 'test'
  max_items = max_items2u = np.inf # 2000 # db_size


results = []
m = tf.keras.metrics.CategoricalAccuracy()
i = 0
all_confusion = np.zeros((params.num_classes, params.num_classes), dtype=np.int)
toshow = '--'
if np.isinf(max_items2u) and csv_name == 'test':
  max_items2u = 153730
for im, lb in tqdm(dataset, desc=csv_name, total=max_items2u / params.BATCH_SIZE, postfix=toshow):
  pred = model.predict(im)
  categories = np.argmax(pred, axis=1)
  if SHOW:
    for gt_, im_, pr in zip(lb, im.numpy(), categories):
      gt = np.argmax(gt_.numpy())
      print(classe_names[gt], classe_names[pr])
      if pr == 0:
        continue
      plt.imshow(im_)#[:, :, ::-1])
      plt.title((gt, pr))
      while not plt.waitforbuttonpress():
        pass
  if csv_name == 'test':
    uuids = [l.numpy().decode("utf-8") for l in lb]
  else:
    m.update_state(lb, pred)
    con_mat = tf.math.confusion_matrix(labels=tf.argmax(lb, axis=1), predictions=tf.argmax(pred, axis=1), num_classes=params.num_classes).numpy()
    all_confusion += con_mat
    toshow += str(m.result().numpy())
    uuids = ['?'] * categories.shape[0]
  for uuid, category in zip(uuids, categories):
    results.append([i, uuid, category])
  i += 1
  if len(results) > max_items:
    break

print('Accuracy: ', m.result().numpy())

with open(model_dir + '/' + csv_name + '.csv', 'w', newline='') as fd:
  fd.write('Id,Predicted\n')
  for r in results:
    fd.write(r[1] + ',' + str(r[2]) + '\n')

if csv_name != 'test':
  idxs2keep = np.where(all_confusion.sum(axis=1) > 0)[0]

  all_confusion = all_confusion[idxs2keep, :][:, idxs2keep]

  all_confusion = all_confusion / all_confusion.sum(axis=1)[:, np.newaxis]

  axis_labels = [classe_names[i] for i in range(len(classe_names)) if i in idxs2keep]

  figure = plt.figure()
  sns.heatmap(all_confusion, annot=True, cmap=plt.cm.Blues, xticklabels=axis_labels, yticklabels=axis_labels)
  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.title(csv_name + ' set')

  plt.show()
