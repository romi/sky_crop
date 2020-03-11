import cv2
import numpy as np
from sklearn import decomposition
from matplotlib import pyplot as plt

# OPEN IMAGE AND EXTRACT ARRAY
im=cv2.imread("/Users/aldo/Desktop/odm_orthophoto.tif",0)
idxs=np.array(np.where(im)).T

# PRINCIPAL COMPONENT ANALYSIS
p=decomposition.PCA()
print(p)
res=p.fit(idxs)
tidxs=res.transform(idxs)
L=np.int(tidxs[:,0].max()-tidxs[:,0].min())
H=np.int(tidxs[:,1].max()-tidxs[:,1].min())
angle=np.arctan(res.components_[0][0]/res.components_[0][1])
print(angle)


# VISUALIZE PCA
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)

targets = ['Iris-setosa']
colors = ['r']

indicesToKeep = p['target'] == target
ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
           , finalDf.loc[indicesToKeep, 'principal component 2']
           , c = color
           , s = 50)
ax.legend(targets)
ax.grid()

# ROTATE IMAGE
im=cv2.imread("/Users/aldo/Desktop/odm_orthophoto.tif")
rot_mat = cv2.getRotationMatrix2D((im.shape[0],im.shape[1]), angle*70/np.pi, 1.0)
# #rot_mat = cv2.getRotationMatrix2D((idxs[:,0].mean(),idxs[:,1].mean()), angle*180/np.pi, 1.0)
result = cv2.warpAffine(im, rot_mat, (2*L,H), flags=cv2.INTER_LINEAR)
# cv2.imshow('rotate',result)
# cv2.waitKey(0)
cv2.imwrite("rot-ortho.png", result)

# idxs=np.array(np.where(result[:,:,0])).T
# print(idxs)
# minx,miny=idxs.min(axis=0)
# maxx,maxy=idxs.max(axis=0)
# # L=maxx-minx
# # H=maxy-miny
# # res=result[minx:maxx,miny:maxy]
# cv2.imwrite("lala.png", res)
