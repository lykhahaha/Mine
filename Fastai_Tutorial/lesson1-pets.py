from os import path
import pandas as pd
import numpy as np
from fastai import vision, metrics

BATCH_SIZE = 64
EPOCHS = 5
PATH_ROOT = path.sep.join(['..', 'datasets', 'oxford-iiit-pet'])
IMAGE_PATHS_ROOT = path.sep.join([PATH_ROOT, 'images'])
ANNO_PATHS = path.sep.join([PATH_ROOT, 'annotations'])

# Get all image names
image_paths = vision.get_image_files(IMAGE_PATHS_ROOT)

# Extract names for image paths
pattern = r'.*images\/(.*)_'
# ImageDataBunch automatically create val set
np.random.seed(42)
data = vision.ImageDataBunch.from_name_re(IMAGE_PATHS_ROOT, image_paths, pattern, ds_tfms=vision.get_transforms(), size=224, bs=BATCH_SIZE).normalize(vision.imagenet_stats)

# Show some samples
print('[INFO] Show sample images...')
data.show_batch(rows=3, figsize=(6, 6))

print(f'[INFO] All classes: {data.classes}')

# Training network
print('[INFO] Training network...')
learner = vision.cnn_learner(data, vision.models.resnet34, metrics=metrics.error_rate)
print('[INFO] model using in text:')
print(learner.model)
learner.fit_one_cycle(EPOCHS)
learner.save('stage-1-34')

# Showing results to check
interp = vision.ClassificationInterpretation.from_learner(learner)
interp.plot_top_losses(9, figsize=(15, 11), heatmap=False)
interp.plot_confusion_matrix(figsize=(12, 12))
interp.most_confused(min_val=2)

# Unfreeze, fine-tuning and learning rates
learner.unfreeze() # train whole model
# learner.load('stage-1')
learner.lr_find()
learner.recoder.plot()
learner.fit_one_cycle(EPOCHS, max_lr=slice(1e-6, 1e-4))
learner.save('stage-1-34')

# Training wih resnet50 to check that we can get higher accuracy
print('[INFO] Training with resnet50...')
data = vision.ImageDataBunch.from_name_re(IMAGE_PATHS_ROOT, image_paths, pattern, ds_tfms=vision.get_transforms(), size=299, bs=BATCH_SIZE//2).normalize(vision.imagenet_stats)

learner = vision.cnn_learner(data, vision.models.resnet50, metrics=metrics.error_rate)

learner.fit_one_cycle(EPOCHS)
learner.unfreeze()
learner.lr_find()
learner.recorder.plot()
learner.fit_one_cycle(EPOCHS, max_slice=(1e-6, 1e-4))
learner.save('stage-1-50')

# Show results to check
interp = vision.ClassificationInterpretation.from_learner(learner)
interp.plot_top_losses(9, figsize=(15, 11), heatmap=False)
interp.plot_confusion_matrix(figsize=(12, 12))
interp.most_confused(min_val=2)

# Training with mnist dataset
print('[INFO] Training with mnist...')
mnist_path = vision.untar_data(vision.URLs.MNIST_SAMPLE)
data = vision.ImageDataBunch.from_folder(mnist_path, ds_tfms=vision.get_transforms(do_flip=False), size=28, bs=BATCH_SIZE)

data.show_batch(3, figsize=(5, 5))

learner = vision.cnn_learner(data, vision.models.resnet18, metrics=metrics.accuracy)
learner.fit(2)

# Training with mnist csv dataset
print('[INFO] Training with mnist csv with various get paths...')
df_path = path.sep.join([mnist_path, 'labels.csv'])
data = vision.ImageDataBunch.from_csv(df_path, ds_tfms=vision.get_transforms(do_flip=False), size=28, bs=BATCH_SIZE)

df = pd.read_csv(df_path)
data = vision.ImageDataBunch.from_df(df, ds_tfms=vision.get_transforms(do_flip=False), size=28)

# Get name fron from_name_func for path: data/mnist_sample/train/3/7463.png
image_paths = [path.sep.join([mnist_path, path]) for path in df['name']]
data = vision.ImageDataBunch.from_name_func(mnist_path, image_paths, ds_tfms=vision.get_transforms(do_flip=False), size=28, label_func = lambda image_path: '3' if f'{path.sep}3{path.sep}' in str(image_path) else '7')
