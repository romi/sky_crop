import cv2
import os
import glob
import numpy as np
from PIL import Image
from _parameters_ import div_x, div_y

# path to save images
img_dir = '/Users/soroush/Desktop/aligned_partitioned_2/*.jpg'
imgs_list = glob.glob(img_dir)
imgs_list.sort()

print("--------------------------------------------")
print("Starting Assembly", '\n')

# rebuilt image from folder
print("Rebuilding image")
images = [Image.open(x) for x in imgs_list]
widths, heights = zip(*(i.size for i in images))

total_width = max(widths) * div_x
total_height = max(heights) * div_y
new_im = Image.new('RGB', (total_width, total_height))

x_offset = 0
y_offset = 0
ind = 0
for j in range(div_y):
    for i in range(div_x):
        im = images[ind]
        new_im.paste(im, (x_offset, y_offset))
        x_offset += im.size[0]
        ind += 1
    y_offset += im.size[1]
    x_offset = 0

# new_im.show('result', new_im)
new_im.save('test.jpg')

print("Finished Assembly!")
print("--------------------------------------------")
