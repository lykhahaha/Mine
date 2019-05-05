from fastai import vision, metrics
from fastai.callback import hooks
from fastai.utils import mem
import numpy as np
from os import path
import torch

vision.defaults.device = vision.defaults.device if torch.cuda.is_available() else torch.device('cpu')

# Download dataset gdrive from Data-Science/datasets

# Download data and get path
fastai_path = vision.Path(path.sep.join(['..', 'datasets', 'camvid-tiramisu']))
PATH = str(fastai_path)
print('CAMVID tiramisu paths:')
print(fastai_path.ls())
BATCH_SIZE = 64
WD = 1e-2
LR = 1e-4
PCT_START_FINETUNE = 0.9 # given the default of 0.3, it means that your LR is going up for 30% of your iterations and then decreasing over the last 70%
PCT_START = 0.8
EPOCHS_FINETUNE = 12
EPOCHS = 12

# Define images and label path
IMAGE_PATH = path.sep.join([PATH, 'train'])

# Define paths of image and label
image_paths = vision.get_image_files(IMAGE_PATH)

# Load some samples to see what's inside
rand_indx = np.random.randint(0, len(image_paths))
sample_image_path = image_paths[rand_indx]
sample_image = vision.open_image(sample_image_path)
sample_image.show(figsize=(6, 6))
# Function to match between image and its label path. E.g. image path: /root/.fastai/data/camvid/images/0006R0_f02910.png; label path: /root/.fastai/data/camvid/labels/0006R0_f02910.png
segment_name_fn = lambda image_path: path.sep.join([f'{image_path.parent}annot', f'{image_path.name}'])
# Load image segmentation by defaults (segment image given in dataset) and vision.open_mask()
sample_label_path = segment_name_fn(sample_image_path)
sample_label = vision.open_image(sample_label_path)
sample_label.show(figsize=(6, 6))
# Note sample segment after preprocess based on vision.open_mask just has 1 depth instead of 3 depth as origin segment
sample_label_preprocessed = vision.open_mask(sample_label_path)
sample_label_preprocessed.show(figsize=(6, 6))
print(sample_label_preprocessed.data) # sample_label_preprocessed is also fastai tensor

# get image dimension (height and width)
image_size = np.array(sample_label_preprocessed.shape[1:])
data_size = image_size//2
objects_in_image = np.array(['Sky', 'Building', 'Pole', 'Road', 'Sidewalk', 'Tree', 'Sign', 'Fence', 'Car', 'Pedestrian', 'Cyclist', 'Void'])

# Determine batch size by gpu free memory to avoid CUDA out pf memory
if torch.cuda.is_available():
    free = mem.gpu_mem_get_free_no_cache()
    if free > 8200:
        BATCH_SIZE = 8
    else:
        BATCH_SIZE = 4
    
    print(f'Using batch size of {BATCH_SIZE}, have {free}MB of GPU RAM free')

# Create dataset
origin_data = vision.SegmentationItemList.from_folder(PATH).split_by_folder(valid='val').label_from_func(segment_name_fn, classes=objects_in_image)
data = origin_data.transform(vision.get_transforms(), tfm_y=True).databunch(bs=BATCH_SIZE).normalize(vision.imagenet_stats)
print(data.show_batch(2, figsize=(10, 7)))
print(data.show_batch(2, figsize=(10, 7), ds_type=vision.DatasetType.Valid))

# Define accuracy
object2id = {value: key for key, value in enumerate(objects_in_image)}
void_index = object2id['Void']

def camvid_accuracy(inputs, target):
    target = target.squeeze(1)
    mask = target != void_index
    return (inputs.argmax(dim=1)[mask] == target[mask]).float().mean()

# Define model
learner = vision.unet_learner(data, vision.models.resnet34, metrics=camvid_accuracy, wd=WD, bottle=True)

# Find good LR
learner.lr_find()
learner.recorder.plot()
learner.fit_one_cycle(EPOCHS_FINETUNE, max_lr=slice(LR), pct_start=PCT_START_FINETUNE)
learner.save('stage-1-34-unet')
# Show results
learner.show_results(rows=3, figsize=(8, 9))

# After warming up, start to train all network
learner.unfreeze()
learner.fit_one_cycle(EPOCHS, max_lr=slice(LR/400, LR/4), pct_start=PCT_START)
learner.save('stage-2-34-unet')