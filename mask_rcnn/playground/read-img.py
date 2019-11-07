# from datetime import datetime import numpy
from exif import Image

with open('test.jpg', 'rb') as image_file:
    print(image_file)
    my_image = Image(image_file)

print(dir(my_image))
print(my_image.datetime)
print(my_image.pixel_x_dimension)
print(my_image.pixel_y_dimension)


#print(datetime.now())
