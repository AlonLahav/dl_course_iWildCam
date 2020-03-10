import numpy as np
import tensorflow as tf

import dataset_exploration
import params


# Get the data
# ------------
root_path_ds = '/media/alonlahav/4T-b/datasets/iwildcam-2019-fgvc6'

def process_path_no_augmentation(label, file_name, location, n_frames):
  fn = root_path_ds + '/train_images/' + file_name
  image = tf.io.read_file(fn)
  image = tf.image.decode_jpeg(image)
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = tf.image.resize(image, (params.IMAGE_SHAPE[0] * params.SPLIT[0], params.IMAGE_SHAPE[0] * params.SPLIT[1]))
  label = tf.one_hot(label, params.num_classes)

  return image, label, location

def process_path(label, file_name, location, n_frames):
  if params.EXPLORE_DATASET and not params.SHOW_IMAGES:
    image = np.zeros(params.IMAGE_SHAPE)
  else:
    fn = root_path_ds + '/train_images/' + file_name
    image = tf.io.read_file(fn)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    if not params.EXPLORE_DATASET:
      image = tf.image.resize(image, (params.IMAGE_SHAPE[0] * params.SPLIT[0], params.IMAGE_SHAPE[0] * params.SPLIT[1]))
  label = tf.one_hot(label, params.num_classes)

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
  if not params.EXPLORE_DATASET:
    dataset = dataset.filter(lambda label, image, location, n_frames: (n_frames == 3))

  if params.SMALL_DATASET:
    dataset = dataset.filter(lambda label, image, location, n_frames: (label == 1 or label == 17 or label == 18))
  dataset = dataset.shuffle(buffer_size=1000)

  if train:
    dataset = dataset.map(process_path)
  else:
    dataset = dataset.map(process_path_no_augmentation)

  if params.EXPLORE_DATASET:
    dataset_exploration.explore_dataset(dataset, num_classes, SHOW_IMAGES)

  dataset = dataset.batch(params.BATCH_SIZE)

  return dataset
