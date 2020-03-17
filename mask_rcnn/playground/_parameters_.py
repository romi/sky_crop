# THIS FILE CENTRALIZES THE PARAMETERS USED FOR THE PROCESS,
# * some parameters are shared between few files.
import os

# Grid size for the cropping of aligned image
div_x = 8
div_y = 3

# base folder: folder name should be the date, and the folder should contain the 'orthomasic.tif'
folder = '/Users/soroush/Desktop/200305'
date = os.path.split(folder)[-1]

# if False: it will only save the output files
view_process = True


# Logs directory for trained model
LOGS_AND_MODEL_DIR = "/Volumes/HDD/Noumena/logs"



"""
to run the pipeline: 

cd path/to/sky_crop/mask_rcnn

python playground/align_ortho.py ; python playground/img_crop.py ; python lettuce_s.py --mode predict ; python playground/img_assemble.py ; python playground/img_indexing.py

"""
