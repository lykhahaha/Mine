# Multiplication
# Each picture has multiple labels. Associated csv file show that each image name is associated to several tags separated by sapces

from fastai import vision, metrics, widgets
import numpy as np
from os import path
import pandas as pd
import torch

PATH = path.sep.join(['..', 'datasets', 'planet'])
EPOCHS = 5
THRESHOLD = 0.2
vision.defaults.device = vision.defaults.device if torch.cuda.is_available() else torch.device('cpu')

# Define transforms
transforms = vision.get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0)

# Due to multi-labeled, ImageList is used rather than ImagedataBunch to make sure the model created has the proper loss function to deal with the multiple classes
np.random.seed(42)
origin_data = vision.ImageList.from_csv(PATH, 'train_v2.csv', folder='train-jpg', suffix='.jpg').split_by_rand_pct(0.2).label_from_df(label_delim=' ')
data = origin_data.transform(transforms, size=256).databunch().normalize(vision.imagenet_stats)
#---------------------OR to set batch size to avoid CUDA out of memory--------------------------------------------
# data = origin_data.transform(transforms, size=256).databunch()
# data.batch_size = 32
# data.normalize(vision.imagenet_stats)
#-------------------------------------------------------------------------------------------------------------------
# Show some sample images, labels are separated by semicolon
print(data.show_batch(rows=3, figsize=(12, 12), heatmap=False))

# In training pass, resnet34 is used again, but we use accuracy_thresh rather than accuracy
arch = vision.models.resnet50
# Define accuracy threshold and f2 score metrics
acc_thresh = vision.partial(metrics.accuracy_thresh, thresh=THRESHOLD)
f2_score = vision.partial(metrics.fbeta, thresh=THRESHOLD)
learner = vision.cnn_learner(data, arch, metrics=[acc_thresh, f2_score])

# Use LR finder to find good lr
learner.lr_find()
learner.recorder.plot()
LR = 0.01
learner.fit_one_cycle(EPOCHS, max_lr=LR)
learner.save('stage-1-34')

# Finetuning model
learner.unfreeze()
# Use LR finder to find good lr
learner.lr_find()
learner.recorder.plot()
learner.fit_one_cycle(EPOCHS, max_lr=slice(1e-6, 1e-4))
learner.save('stage-2-34')

# Export for submission
learner.export()
# Test on testset images
test_data = vision.ImageList.from_folder(path.sep.join([PATH, 'test-jpg'])).add(vision.ImageList.from_folder(path.sep.join([PATH, 'test-jpg-additional'])))
learner = vision.load_learner(PATH, test=test_data)
preds, _ = learner.get_preds(ds_type=vision.DatasetType.Test) # _ means targets, we don't need it

# Prepare for writing dataframe
test_labels = []
for pred in preds:
    text_label = []
    for idx, pred_val in enumerate(pred):
        if pred_val > THRESHOLD:
            text_label.append(learner.data.classes[idx])
    test_labels.append(' '.join(text_label))

test_data_names = []
for file_name in learner.data.test_ds.items:
    test_data_names.append(file_name.name[:-4])

# Construct dataframe
df = pd.DataFrame({
    'image_name': test_data_names,
    'tags': test_labels
}, columns=['image_name', 'tags'])
df.to_csv(path.sep.join([PATH, 'submission.csv']), index=False)
# kaggle competitions submit planet-understanding-the-amazon-from-space -f {path.sep.join([PATH, 'submission.csv'])} -m "Submit 9.25 27/4"