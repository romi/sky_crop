import cv2
assert cv2.__version__[0] == '3', 'The fisheye module requires opencv version >= 3.0.0'
import numpy as np
import os
from imutils import paths
import glob
import sys

# You should replace these 3 lines with the output in calibration step
DIM=(2304, 1536)
K=np.array([[1559.8270222529836, 0.0, 1133.0908373820648], [0.0, 1565.847155381092, 783.6060590122621], [0.0, 0.0, 1.0]])
D=np.array([[-0.07070727300518695], [-0.02487167453074337], [-0.08166276318367528], [0.5478459936094181]])


img_path = sorted(list(paths.list_images("/Users/aldo/Downloads/image-stitching-opencv/images/romi/")))
output = ("/Users/aldo/PycharmProjects/cloud-voxelization/romi/und_imgs")

print(img_path[0])

for j in range(0,len(img_path)):
    img = cv2.imread(img_path[j])
    print(j)
    h,w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    # cv2.imshow("undistorted", undistorted_img)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(os.path.join(output, "UNDIMG"+(str(j))+".jpg"), undistorted_img)
