import cv2
import os
import glob
import numpy as np
from PIL import Image

#open image
path = '/Users/aldo/Desktop/balloon.jpeg'
img = cv2.imread(path)
img_name = os.path.splitext(os.path.split(path)[-1])[0]

print(img_name)
print(img.shape)
cv2.imshow('img', img)

#get section number and xy limits
img_sects = int(img.shape[1]/400)
limY = int(img.shape[0])
limX = int(img.shape[1]/img_sects)


#path to save images
img_dir = '/Users/aldo/Desktop/path_img'
imgs_list = []

#crop and save images
for i in range(0,img_sects):
    if i == 0:
        crop_img = img[0:limY, 0:limX]
    else :
        crop_img = img[0:limY, limX*i:limX*(i+1)]

    img_file = img_name + '_' + str(i) + '.jpg'
    # print(img_file)

    cv2.imshow('crop',crop_img)
    img_path = os.path.join(img_dir,img_file)
    imgs_list.append(img_path)
    cv2.imwrite(os.path.join(img_dir,img_file), crop_img)
    cv2.waitKey(0)


#rebuilt image from folder
print(imgs_list)
images = [Image.open(x) for x in imgs_list]
widths, heights = zip(*(i.size for i in images))

total_width = sum(widths)
max_height = max(heights)

new_im = Image.new('RGB', (total_width, max_height))

x_offset = 0
for im in images:
  new_im.paste(im, (x_offset,0))
  x_offset += im.size[0]

new_im.show('result',new_im)
new_im.save('test.jpg')



