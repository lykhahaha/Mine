from fastai import vision
import numpy as np
from os import path

# Download data and get path
fastai_path = vision.untar_data(vision.URLs.BIWI_HEAD_POSE)
PATH = str(fastai_path)
print('BIWI_HEAD_POSE paths:')
print(fastai_path.ls())
BATCH_SIZE = 64
WD = 1e-2
LR = 2e-2
PCT_START_FINETUNE = 0.9 # given the default of 0.3, it means that your LR is going up for 30% of your iterations and then decreasing over the last 70%
PCT_START = 0.8
EPOCHS = 5

# Default value from dataset
RGB_CAL = np.genfromtxt(path.sep.join([PATH, '01', 'rgb.cal']), skip_footer=6)
print('[INFO] RGB cal:')
print(RGB_CAL)

# define function to match between image path and text file path. E.g: image path: /root/.fastai/data/biwi_head_pose/01/frame_00003_rgb.jpg; text file name: /root/.fastai/data/biwi_head_pose/01/frame_00003_pose.txt
image_path2text_file_name = lambda image_path: path.sep.join([f'{str(image_path)[:-7]}pose.txt'])

# Sample image
image_name = path.sep.join(['01', 'frame_00003_rgb.jpg'])
image_path = path.sep.join([PATH, image_name])
sample_image = vision.open_image(image_path)
sample_image.show(figsize=(6, 6))
# Load center point of this image from text file in dataset
center_pt = np.genfromtxt(image_path2text_file_name(image_path), skip_header=3)
print(f'[INFO] center point of sample image: {center_pt}')

# define function
def convert_biwi(coords):
    c1 = coords[0] * RGB_CAL[0][0]/coords[2] + RGB_CAL[0][2]
    c2 = coords[1] * RGB_CAL[1][1]/coords[2] + RGB_CAL[1][2]
    return vision.tensor([c2, c1])

def get_center_pt(image_path):
    center_pt = np.genfromtxt(image_path2text_file_name(image_path), skip_header=3)
    return convert_biwi(center_pt)

def get_image_pt(image, points):
    return vision.ImagePoints(vision.FlowField(image.size, points), scale=True)

converted_center_pt = get_center_pt(image_path)
print(f'[INFO] center point of sample image after converted: {converted_center_pt}')
sample_image.show(y=get_image_pt(sample_image, converted_center_pt), figsize=(6, 6))

# Create dataset
# Note: image_path.parent.name to get '01' from '/root/.fastai/data/biwi_head_pose/01/frame_00003_rgb.jpg'
data_size=(120, 160)
data = vision.PointsItemList.from_folder(PATH).split_by_valid_func(lambda image_path: image_path.parent.name=='13').label_from_func(get_center_pt).transform(vision.get_transforms(), tfm_y=True, size=data_size).databunch().normalize(vision.imagenet_stats)
print('[INFO] some sample images from loaded dataset')
data.show_batch(3, figsize=(9, 6))

# Define model
learner = vision.cnn_learner(data, vision.models.resnet34)
# Find good LR
learner.lr_find()
learner.recorder.plot()
learner.fit_one_cycle(EPOCHS, max_lr=slice(LR))
learner.save('stage-1')
learner.show_results()

# More Data augmentation, instead of default transforms
tfms = vision.get_transforms(max_rotate=20, max_zoom=1.5, max_lighting=0.5, max_warp=0.4, p_affine=1, p_lighting=1)
data = vision.PointsItemList.from_folder(PATH).split_by_valid_func(lambda image_path: image_path.parent.name=='13').label_from_func(get_center_pt).transform(tfms, tfm_y=True, size=data_size).databunch().normalize(vision.imagenet_stats)
# train again with defined augmentation
learner.lr_find()
learner.recorder.plot()
learner.fit_one_cycle(EPOCHS, max_lr=slice(LR))
learner.save('stage-2')
learner.show_results()
def _plot(i, j, ax):
    x, y = data.train_ds[0]
    x.show(ax, y=y)
vision.plot_multi(_plot, 3, 3, figsize=(8, 6))