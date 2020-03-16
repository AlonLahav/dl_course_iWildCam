# dl_course_iWildCam
Solution For kaggle challange for iWildCam 2019

# Already done:
- Used <a href="https://github.com/tensorflow/docs/blob/master/site/en/tutorials/images/transfer_learning_with_hub.ipynb">TF example</a>
- Dataset was downloaded and extracted
- Trained over this dataset
- Used it on our data (inference then train / fine-tune)
- Split to train / validate
- Used spacial feature extraction
- Evaluate on train / validate / test set
- Submission with dummy file, just to check the format
- Submit to Kaggl

## ToDo

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
Change the `run` variable to the current run folder.

Set dataset to the desired one. 

The results (CSV file, for submition) should be at the run folder.
```
python evaluate.py
```


## Submit to Kaggle
Go to submission page at Kaggle and drop the CSV file.