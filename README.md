# dl_course_iWildCam
Solution For kaggle challange for iWildCam 2019

# Already done:
- Used <a href="https://github.com/tensorflow/docs/blob/master/site/en/tutorials/images/transfer_learning_with_hub.ipynb">TF example</a>
- Dataset was downloaded and extracted
- Trained over this dataset
- Used it on our data (inference then train / fine-tune)
- Split to train / validate

## ToDo
- Evaluate on train / validate / test set
- Submit to Kaggle
- Check that all works
- Fix train data for tensorboard
- Split to 4 x 3 slices
- Check notebooks and pickup some ideas

## Data
Go to <a href="https://www.kaggle.com/c/iwildcam-2019-fgvc6/data">Kaggle page</a>.
Download `iwildcam-2019-fgvc6.zip` file (46.6G).

## Installations
Using TensorFlow 2.1, and tensorflow_hub.
Other pakedges:
- seaborn
- matplotlib

## Notes about TensorFlow Hub
TF hub page: <a href="https://www.tensorflow.org/hub">link</a>.

## Train
```
python train_val.py
```

## Test
```
python evaluate.py
```

## Submit to Kaggle

