import cv2
import numpy as np
from sklearn import decomposition
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

# OPEN IMAGE AND EXTRACT ARRAY
im=cv2.imread("/Users/aldo/Desktop/odm_orthophoto.tif",0)
idxs=np.array(np.where(im)).T

# PRINCIPAL COMPONENT ANALYSIS
p=decomposition.PCA()
res=p.fit(idxs)
print(idxs)
print(p.components_)
print(p.explained_variance_)

# VISUALIZE PCA
def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)

# PLOT DATA
plt.scatter(idxs[:, 0], idxs[:, 1], s=0.05, alpha=0.01)
for length, vector in zip(p.explained_variance_, p.components_):
    v = vector * 1 * np.sqrt(length)
    draw_vector(p.mean_, p.mean_ + v)
plt.axis('equal')
plt.show()

# tidxs=res.transform(idxs)
# L=np.int(tidxs[:,0].max()-tidxs[:,0].min())
# H=np.int(tidxs[:,1].max()-tidxs[:,1].min())
# angle=np.arctan(res.components_[0][0]/res.components_[0][1])
# print(angle)
#
#
# # ROTATE IMAGE
# im=cv2.imread("/Users/aldo/Desktop/odm_orthophoto.tif")
# rot_mat = cv2.getRotationMatrix2D((im.shape[0],im.shape[1]), angle*70/np.pi, 1.0)
# # #rot_mat = cv2.getRotationMatrix2D((idxs[:,0].mean(),idxs[:,1].mean()), angle*180/np.pi, 1.0)
# result = cv2.warpAffine(im, rot_mat, (2*L,H), flags=cv2.INTER_LINEAR)
# # cv2.imshow('rotate',result)
# # cv2.waitKey(0)
# cv2.imwrite("rot-ortho.png", result)

# idxs=np.array(np.where(result[:,:,0])).T
# print(idxs)
# minx,miny=idxs.min(axis=0)
# maxx,maxy=idxs.max(axis=0)
# # L=maxx-minx
# # H=maxy-miny
# # res=result[minx:maxx,miny:maxy]
# cv2.imwrite("lala.png", res)
