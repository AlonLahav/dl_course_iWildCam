
import pylab as plt
import numpy as np

import dataset_utils
import params

classes = """empty, 0
deer, 1
moose, 2
squirrel, 3
rodent, 4
small_mammal, 5
elk, 6
pronghorn_antelope, 7
rabbit, 8
bighorn_sheep, 9
fox, 10
coyote, 11
black_bear, 12
raccoon, 13
skunk, 14
wolf, 15
bobcat, 16
cat, 17
dog, 18
opossum, 19
bison, 20
mountain_goat, 21
mountain_lion, 22""".split('\n')

classe_names = [i.split(', ')[0] for i in classes]

def explore_dataset(dataset, n_classes, show_images):
  show_dataset(dataset)
  n_locations = 150
  dataset_size = 0
  items_per_class = np.zeros((n_classes,))
  items_per_location = np.zeros((n_locations,))

  if show_images:
    plt.figure()
    i = 1
    cls_show = np.zeros((n_classes, ))

  for im, label, location in dataset:
    dataset_size += 1
    l = np.where(label)[0][0]
    items_per_class[l] += 1
    items_per_location[location] += 1
    if show_images and not cls_show[l]:
      cls_show[l] = 1
      plt.subplot(2, 2, cls_show.sum())
      plt.title(classe_names[l])
      plt.imshow(im)
    if show_images and cls_show.sum() >= 3:
      break

  print('dataset_size: ', dataset_size)

  plt.figure()
  ind = np.arange(n_classes)
  p = plt.barh(ind[1:], items_per_class[1:])
  plt.yticks(ind[1:], classe_names[1:])
  plt.xlabel('# of Empty class: ' + str(items_per_class[0]))
  plt.title('Number of occurances per class in the Dataset\nTotal dataset size: ' + str(dataset_size))

  plt.figure()
  ind_l = np.arange(n_locations)
  plt.bar(ind_l, items_per_location)
  plt.title('Items per location')

  plt.show()


def show_dataset(dataset):
  for images, labels in dataset:
    image = images[0]
    label = labels[0]
    # print(image.shape)
    if image.shape[2] == 3:
      # continue
      # if location != 33:
      #  continue
      plt.imshow(image)
      plt.title(label.numpy())
      while not plt.waitforbuttonpress():
        pass

if __name__ == '__main__':
  params.BATCH_SIZE = 1
  if 0:  # Validate
    dataset, _ = dataset_utils.get_dataset(train=False)
  elif 1:  # Train
    dataset, _ = dataset_utils.get_dataset(train=True)
  else:  # Test, 153730 rows. This file should have a header row. CSV: integer-index , UUID, integer-result
    dataset, _ = dataset_utils.get_dataset_kaggle_test()

  show_dataset(dataset)