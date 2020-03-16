import cv2
import os
import glob
import numpy as np
from PIL import Image

# open image
path = '/Users/soroush/Desktop/aligned.png'
img = cv2.imread(path)
img_name = os.path.splitext(os.path.split(path)[-1])[0]
height, width = img.shape[0:2]

print("--------------------------------------------")
print("name\t{0}".format(img_name))
print("height\t{0}".format(height))
print("width\t{0}".format(width))

# create a grid to partition the image
div_x = 10
div_y = 3
print("div X\t{0}".format(div_x))
print("div Y\t{0}".format(div_y))

limX = int(width / div_x)
limY = int(height / div_y)
print("subX\t{0}".format(limX))
print("subY\t{0}".format(limY))

# path to save images
img_dir = '/Users/soroush/Desktop/aligned_partitioned/'
imgs_list = []

# crop and save images
for j in range(div_y):
    for i in range(div_x):
        crop_img = img[limY*j:limY*(j+1), limX*i:limX*(i+1)]
        img_file = img_name + '_' + str(j) + '_' + str(i) + '.jpg'
        img_path = os.path.join(img_dir, img_file)
        imgs_list.append(img_path)
        cv2.imwrite(os.path.join(img_dir, img_file), crop_img)
        # cv2.imshow(img_file, crop_img)
        # cv2.waitKey(0)


print ("Finished partitioning!")

# rebuilt image from folder
print ("Rebuilding image")
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

new_im.show('result', new_im)
new_im.save('test.jpg')

print("--------------------------------------------")