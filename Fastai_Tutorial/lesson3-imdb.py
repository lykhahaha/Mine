from fastai import vision
import numpy as np
import pandas as pd
from os import path

# Download data and get path
fastai_path = text.untar_data(text.URLs.IMDB_SAMPLE)
PATH = str(fastai_path)
print('IMDB_SAMPLE paths:')
print(fastai_path.ls())
BATCH_SIZE = 64
WD = 1e-2
LR = 2e-2
PCT_START_FINETUNE = 0.9 # given the default of 0.3, it means that your LR is going up for 30% of your iterations and then decreasing over the last 70%
PCT_START = 0.8
EPOCHS = 5

# load csv file including label and text
df = pd.read_csv(path.sep.join([PATH, 'texts.csv']))
print(f'[INFO] Sample text: {df['text'][3]}')