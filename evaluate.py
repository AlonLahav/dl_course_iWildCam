
import tensorflow as tf

import dataset_utils

model_dir = '/home/alonlahav/git-projects/dl_course_iWildCam/10.03.2020..22.31/model/mymodel_1/'

reloaded = tf.keras.models.load_model(model_dir)

test_dataset = dataset_utils.get_dataset(locations=[0, 1, 2], train=False)

image_batch, labels, locations = iter(test_dataset).next()
reloaded_result_batch = reloaded.predict(image_batch)
print(reloaded_result_batch)
