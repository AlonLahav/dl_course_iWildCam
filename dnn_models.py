import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras import layers

import params
import dataset_utils


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


def get_feature_extractor_layer():
  feature_extractor_layer = hub.KerasLayer(params.feature_extractor_url,
                                           input_shape=(224, 224, 3))
  feature_extractor_layer.trainable = True

  return feature_extractor_layer

def get_feature_extractor_layer_spacial():
  model = tf.keras.applications.MobileNetV2(include_top=False,
                                            weights='imagenet')

  return model


def get_2x2_model():
  feature_extractor_layer = get_feature_extractor_layer()
  inputs = keras.Input(shape=(params.IMAGE_SHAPE[0] * params.SPLIT[0], params.IMAGE_SHAPE[1] * params.SPLIT[1], 3))
  x1 = feature_extractor_layer(inputs[:, :params.IMAGE_SHAPE[0],                        :params.IMAGE_SHAPE[1]]                )
  x2 = feature_extractor_layer(inputs[:, params.IMAGE_SHAPE[0]:params.IMAGE_SHAPE[0]*2, :params.IMAGE_SHAPE[1]]                )
  x3 = feature_extractor_layer(inputs[:, :params.IMAGE_SHAPE[0],                        params.IMAGE_SHAPE[1]:params.IMAGE_SHAPE[1]*2])
  x4 = feature_extractor_layer(inputs[:, params.IMAGE_SHAPE[0]:params.IMAGE_SHAPE[0]*2, params.IMAGE_SHAPE[1]:params.IMAGE_SHAPE[1]*2])
  pooling = (x1 + x2 + x3 + x4) / 4
  outputs = layers.Dense(params.num_classes)(pooling)
  model = keras.Model(inputs=inputs, outputs=outputs, name='classification')

  return model


def get_vanila_model():
  feature_extractor_layer = get_feature_extractor_layer()

  model = tf.keras.Sequential([
    feature_extractor_layer,
    layers.Dense(params.num_classes)
  ])
  return model


def get_spacial_av_pooling_model():
  feature_extractor_layer = get_feature_extractor_layer_spacial()

  model = tf.keras.Sequential([
    feature_extractor_layer,
    tf.keras.layers.GlobalAveragePooling2D(),
    layers.Dense(params.num_classes)
  ])
  return model


def get_model():
  if params.MODEL2USE == 'SPLIT_2x2':
    model = get_2x2_model()
  elif params.MODEL2USE == 'VANILA':
    model = get_vanila_model()
  elif params.MODEL2USE == 'SPATIAL':
    model = get_spacial_av_pooling_model()
  return model


def check():
  test_dataset, db_size = dataset_utils.get_dataset(locations=params.test_locations, train=False)

  model = get_spacial_av_pooling_model()

  for im, lbl in test_dataset:
    p = model(im)


if __name__ == '__main__':
  check()