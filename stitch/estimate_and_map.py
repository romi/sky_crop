import os
import match_images as mi
import numpy as np
import time
import cv2

def estimate(imdir):
   fs=os.listdir(imdir)
   fs.sort()

   filter_params={'x1'   : 0,
                  'x2'   : 10000,
                  'ymin' : 0,
                  'xmin1' :0,
                  'xmax1' :10000,
                  'xmin2' :0,
                  'xmax2' : 10000,
                  'dxmin': 0,
                  'dymin': 0,
                  'dxmax': 10000,
                  'dymax': 10000,
                  'match_ratio':.5
             }     
   

   resdir=imdir+"../stitch"

   if not(os.path.exists(resdir)):
   	print
   	os.makedirs(resdir)

   for i in range(len(fs)-1):
      mi.estim(fs[i], fs[i+1], imdir, filter_params, resdir,True)

def trComb(t1,t2):
    res=np.zeros([2,3])
    res[:2,:2]= np.dot(t2[:2,:2],t1[:2,:2])
    res[:,2]=np.dot(t2[:2,:2],t1[:,2])+t2[:,2]
    return res

def trPoint(t,x):
    return np.dot(t[:2,:2],x.T).T+t[:2,2]

def getTrs(w, h, fs, K, M, resdir):
   idtr=np.zeros([2,3])
   idtr[0,0]=1
   idtr[1,1]=1
   trs=[idtr]
   Hs=[idtr]
   pts=np.array([[0,0],[w,0],[w,h],[0,h],[0,0]])

   for i in range(M): 
    Hs.append(np.load(resdir+"/HR_%s_%s.npy"%(fs[K*i],fs[K*(i+1)]))[0])
   for i in range(M):
      tr_all=idtr
      for j in range(i+1,M+1): 
        tr_all=trComb(tr_all,Hs[j])
      trs.append(tr_all)
      pts=np.concatenate([pts,trPoint(trs[-1],pts[:5])])
   trs.append(idtr)
   return trs, pts


def map(imdir):
   fs=os.listdir(imdir)
   fs.sort()

   i=0
   M=len(fs)-1
   cols=np.random.rand(M+1,3)

   K=1
   N=len(fs)/K
   resdir=imdir+"../stitch/"

   im0=cv2.imread(imdir+fs[0])
   h, w, _= im0.shape
   trs, pts=getTrs(w, h, fs, 1, M, resdir)

   Wmax,Hmax = np.max(pts,axis=0).astype(np.int)+1
   Wmin,Hmin = np.min(pts,axis=0).astype(np.int)
   Wres= Wmax-Wmin
   Hres= Hmax-Hmin

   offset=np.zeros([2,3])
   offset[0,0]=1
   offset[1,1]=1
   offset[:,2]=[-Wmin,-Hmin]

   res=np.zeros([Hres,Wres,3],dtype=np.uint8)

   for i in range(1,M):
      idx=fs[i]
      im1=cv2.resize(cv2.imread(imdir+idx),(w,h))

      warped=cv2.warpAffine(im1, trComb(trs[i+1],offset), (Wres,Hres))
      imin=int(np.where(warped)[0].min()+.2*h)
      imax=int(imin+.6*h)
      res[imin:imax]=warped[imin:imax]

   cv2.imwrite(resdir+"map.png",res)


t0=time.time()

imdir="images/"
estimate(imdir)
map(imdir)

print(time.time()-t0)

