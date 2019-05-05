# DATASETS: crawl from images.google.com
# datasets/bears/black; datasets/bears/grizzly, datasets/bears/teddys

from os import path
import numpy as np
import pandas as pd
from fastai import vision, metrics, widgets
import torch

PATH = path.sep.join(['..', 'datasets', 'bears'])
BATCH_SIZE=64
EPOCHS = 5
vision.defaults.device = vision.defaults.device if torch.cuda.is_available() else torch.device('cpu')

#---------------------------------Construct datasets from images.google.com---------------------------------------
# Get a list of URLs
# Go to Google Images and search for the images you are interested in. The more specific you are in your Google Search, the better the results and the less manual pruning you will have to do.

# Scroll down until you've seen all the images you want to download, or until you see a button that says 'Show more results'. All the images you scrolled past are now available to download. To get more, click on the button, and continue scrolling. The maximum number of images Google Images shows is 700.

# It is a good idea to put things you want to exclude into the search query, for instance if you are searching for the Eurasian wolf, "canis lupus lupus", it might be a good idea to exclude other variants:

# "canis lupus lupus" -dog -arctos -familiaris -baileyi -occidentalis

# You can also limit your results to show only photos by clicking on Tools and selecting Photos from the Type dropdown

# Download into file

# Now you must run some Javascript code in your browser which will save the URLs of all the images you want for you dataset.

# Press CtrlShiftJ in Windows/Linux and CmdOptJ in Mac, and a small window the javascript 'Console' will appear. That is where you will paste the JavaScript commands.

# You will need to get the urls of each of the images. You can do this by running the following commands:

# urls = Array.from(document.querySelectorAll('.rg_di .rg_meta')).map(el=>JSON.parse(el.textContent).ou); window.open('data:text/csv;charset=utf-8,' + escape(urls.join('\n')));

classes = ['black', 'teddys', 'grizzly']
for species in classes:
    vision.download_images(species, path.sep.join([PATH, species]))

# remove all images cannot open
for species in classes:
    vision.verify_images(path.sep.join([PATH, species]), delete=True)

np.random.seed(42)
data = vision.ImageDataBunch.from_folder(PATH, train='.', valid_pct=0.2, ds_tfms=vision.get_transforms(), size=224, bs=BATCH_SIZE, num_workers=4).normalize(vision.imagenet_stats)

# Show some samples
print('[INFO] Show sample images...')
data.show_batch(rows=3, figsize=(6, 6))

print(f'[INFO] All classes: {data.classes}')
print(f'[INFO] Traning images info:')
print(data.train_ds)

# Training network
print('[INFO] Training network...')
learner = vision.cnn_learner(data, vision.models.resnet34, metrics=metrics.accuracy)
learner.fit_one_cycle(EPOCHS)
# Plot loss for understanding
learner.recorder.plot_losses()
learn.save('stage-1-34')

# Unfreeze, fine-tuning and learning rates
learner.unfreeze()
learner.lr_find()
learner.recorder.plot()
learner.fit_one_cycle(EPOCHS, max_lr=slice(1e-6, 1e-4))
learner.recorder.plot_losses()
learner.save('stage-2-34')

# Showing results to check
interp = vision.ClassificationInterpretation.from_learner(learner)
interp.plot_top_losses(9, figsize=(12, 12), heatmap=False)
interp.plot_confusion_matrix()
interp.most_confused(min_val=2)

# Cleaning validation dataset b/c some images are not in any classes
# In order to clean the entire set of images, we need to create a new dataset without the split
data_for_cleaner = vision.ImageList.from_folder(PATH).split_none().label_from_folder().transform(vision.get_transforms(), size=224).databunch()
# create a new learner to use our new databunch with all the images.
learner_clean = vision.cnn_learner(data_for_cleaner, vision.models.resnet34, metrics=metrics.accuracy)
learner_clean.load('stage-2-34')

# Get losses and indexes from top_losses
data_for_cleaner, indexes = widgets.DatasetFormatter.from_toplosses(learner_clean, n_imgs=50, ds_type=vision.DatasetType.Train)
widgets.ImageCleaner(data_for_cleaner, indexes, PATH) # create cleaned.csv which matches with your cleaner
# Load data with cleaned.csv
df = pd.read_csv(path.sep.join([PATH, 'cleaned.csv']))
data = vision.ImageList.from_df(df, PATH).split_none().label_from_folder().transform(vision.get_transforms(), size=224).databunch()
# -------------------------OR--------------------
# find duplicates and remove
data_for_cleaner, indexes = widgets.DatasetFormatter.from_similars(learner_clean)
widgets.ImageCleaner(data_for_cleaner, indexes, PATH, duplicates=True)

# Recreate your ImageDataBunch from your cleaned.csv to include the changes you made in your data!
df = pd.read_csv(path.sep.join([PATH, 'cleaned.csv']))
data = vision.ImageDataBunch.from_df(PATH, df, ds_tfms=vision.get_transforms(), size=224, bs=BATCH_SIZE, num_workers=4).normalize(vision.imagenet_stats)


# Put model to production
# .export() will create export.pkl that contains everything we need to deploy our model (the model, the weights but also some metadata like the classes or the transforms/normalization used)
learner.export()
test_image = vision.open_image('your/path/to/bear/image')
learner = vision.load_learner(PATH)
pred_class, pred_idx, pred_value = learner.predict(test_image)