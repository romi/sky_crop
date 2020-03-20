import cv2
import os
import glob
import numpy as np
from PIL import Image
from _parameters_ import div_x, div_y, date, folder, view_process

# path to read images
img_dir = '{}/detected/*.jpg'.format(folder)
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

if view_process:
    new_im.show('Rebuilt Image', new_im)

new_im.save('{0}/{1}_detected_total.jpg'.format(folder, date))

print("Finished Rebuilding!")
print("--------------------------------------------")
