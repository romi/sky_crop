# THIS FILE CENTRALIZES THE PARAMETERS USED FOR THE PROCESS,
# * some parameters are shared between few files.
import os

# Grid size for the cropping of aligned image
div_x = 8
div_y = 3

# base folder: folder name should be the date, and the folder should contain the 'orthomasic.tif'
folder = '/Users/soroush/Desktop/200320'
date = os.path.split(folder)[-1]

# if False: it will only save the output files
view_process = False


# Logs directory for trained model
LOGS_AND_MODEL_DIR = "/Volumes/HDD/Noumena/logs"


"""
to run the pipeline: 

cd path/to/sky_crop/mask_rcnn

python demo/01_align.py ; python demo/02_crop.py ; python demo/03_detection.py --mode predict ; python demo/04_assemble.py ; python demo/05_indexing.py

"""
