import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras import layers

import params


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
  feature_extractor_layer.trainable = False

  return feature_extractor_layer


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