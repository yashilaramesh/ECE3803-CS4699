# Kaggle dataset: https://www.kaggle.com/datasets/quadeer15sh/image-super-resolution-from-unsplash
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import imread
from PIL import Image
import os
image_path = os.path.join(os.path.dirname(__file__), 'images', '134.jpg')
image_row = np.array(imread(image_path))
print(image_row.shape)

img = Image.open(image_path)
width, height = img.size
block_size = 8
image_blocks = []
for y in range(0, height, block_size):
    for x in range(0, width, block_size):
        box = (x, y, x + block_size, y + block_size)
        block = img.crop(box)
        image_blocks.append(block)
