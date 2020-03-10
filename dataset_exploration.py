
import pylab as plt
import numpy as np

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

SHOW_IMAGES = True

def explore_dataset(dataset, n_classes, show_images):
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
      plt.subplot(3, 4, cls_show.sum())
      plt.title(classe_names[l])
      plt.imshow(im)
    if show_images and cls_show.sum() >= 12:
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

  exit(0)