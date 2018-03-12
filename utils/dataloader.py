####DATASET ETL HERE####

import os
import numpy as np
import random
import pickle
from PIL import Image
from functools import partial

def normalize(x):
    return x / 255.0

def split_train_val(dataset, val_percent=0.05):
    dataset = list(dataset)
    length = len(dataset)
    n = int(length * val_percent)
    random.seed(42)
    random.shuffle(dataset)
    return {'train': dataset[:-n], 'val': dataset[-n:]}

def resize_and_crop(pilimg, scale=0.5, final_height=800):
    w = pilimg.size[0]
    h = pilimg.size[1]
    newW = int(w * scale)
    newH = int(h * scale)

    if not final_height:
        diff = 0
    else:
        diff = newH - final_height

    img = pilimg.resize((newW, newH))
    img = img.crop((0, 0, newW, final_height))
    return img, h, w


def batch(iterable, batch_size):
    """Yields lists by batch"""
    b = []
    for i, t in enumerate(iterable):
        b.append(t)
        if (i+1) % batch_size == 0:
            yield b
            b = []
    if len(b) > 0:
        yield b

