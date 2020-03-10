
import tensorflow as tf

model_dir = '/home/alonlahav/git-projects/dl_course_iWildCam/10.03.2020..22.31/model/mymodel_1/saved_model.pb'

reloaded = tf.keras.models.load_model(model_dir)
reloaded_result_batch = reloaded.predict(image_batch)
