import torch
from torch import nn
import torch.nn.functional as F
import re
import os
import unicodedata
import numpy s np

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

MAX_LENGTH = 10 # Maximum sentence length
PAD_TOKEN = 0 # Used for padding short sentences
SOS_TOKEN = 1 # Start-of-sentence token
EOS_TOKEN = 2 # End-of-sentence token

