import numpy as np
import tensorflow as tf

import dataset_exploration
import params


root_path_ds = '/media/alonlahav/4T-b/datasets/iwildcam-2019-fgvc6'
train_csv_file = root_path_ds + '/train.csv'
test_csv_file = root_path_ds + '/test.csv'

def process_path_test(file_name, uuid):
  fn = root_path_ds + '/test_images/' + file_name
  image = tf.io.read_file(fn)
  image = tf.image.decode_jpeg(image)
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = image[12:]
  image = tf.image.resize(image, (params.IMAGE_SHAPE[0], params.IMAGE_SHAPE[0]))

  return image, uuid


def process_path_no_augmentation(label, file_name, location=0, n_frames=0):
  fn = root_path_ds + '/train_images/' + file_name
  image = tf.io.read_file(fn)
  image = tf.image.decode_jpeg(image)
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = image[12:]
  image = tf.image.resize(image, (params.IMAGE_SHAPE[0], params.IMAGE_SHAPE[0]))
  label = tf.one_hot(label, params.num_classes)

  return image, label #, location

def process_path(label, file_name, location, n_frames):
  image, label = process_path_no_augmentation(label, file_name, location, n_frames)

  # data augmentation
  image = tf.image.random_flip_left_right(image)
  #image = tf.image.random_hue(image, 0.1)
  #image = tf.image.random_brightness(image, 0.1)
  #image = tf.image.random_contrast(image, 0.85, 1.15)

  return image, label

def get_dataset_kaggle_test():
  record_defaults = ['jpg', 'uuid']
  select_cols = [1, 3] # file=name = 1, uuid = 3
  dataset = tf.data.experimental.CsvDataset(test_csv_file, record_defaults, select_cols=select_cols, header=True) # category-id / file name / location
  #dataset = dataset.shuffle(buffer_size=1000)
  #dataset = dataset.skip(80000)
  dataset = dataset.map(process_path_test)
  dataset = dataset.batch(params.BATCH_SIZE)
  return dataset, 153730


def get_dataset(train=True):
  record_defaults = [-1, '--', -1, -1]

  csv_file = train_csv_file
  select_cols = [0, 2, 5, 8]
  dataset = tf.data.experimental.CsvDataset(csv_file, record_defaults, select_cols=select_cols, header=True) # category-id / file name / location
  dataset = dataset.filter(lambda label, image, location, n_frames: (n_frames == 3))
  if params.SMALL_DATASET:
    dataset = dataset.filter(lambda label, image, location, n_frames: (label == 1 or label == 17 or label == 18))
  dataset = dataset.shuffle(buffer_size=1000)
  db_size_ = len(list(dataset))

  train_size = int(db_size_ * 0.9)
  if train:
    dataset = dataset.take(train_size)
    dataset = dataset.map(process_path)
    db_size = train_size
  else:
    dataset = dataset.skip(train_size)
    dataset = dataset.map(process_path_no_augmentation)
    db_size = db_size_ - train_size

  dataset = dataset.batch(params.BATCH_SIZE)

  return dataset, db_size
