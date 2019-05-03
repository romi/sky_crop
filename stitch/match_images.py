import numpy as np
import scipy as sp
import scipy.signal
import cv2
import time
import os
import json

def exgreen(im_BGR, cvtype=False):
   Ms=np.max(im_BGR,axis=(0,1)).astype(np.float) 
   im_Norm=im_BGR/Ms
   L=im_Norm.sum(axis=2)
   res = 3*im_Norm[:,:,1]/(L)-1
   if cvtype:
      M=res.max()
      m=res.min()
      res = (255*(res-m)/(M-m)).astype(np.uint8)
   return res

def get_flannkd():
    FLANN_INDEX_KDTREE=1
    flann_pars=dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_pars=dict(checks=100)
    flann=cv2.FlannBasedMatcher(flann_pars, {})# search_pars)
    return flann

def get_flannlsh():
    FLANN_INDEX_LSH=6
    flann_pars= dict(algorithm = FLANN_INDEX_LSH,
                       table_number = 6, # 12
                       key_size = 12,     # 20
                       multi_probe_level = 1) #2

    search_pars=dict(checks=100)
    flann=cv2.FlannBasedMatcher(flann_pars, {})# search_pars)
    return flann

def getImageCorners(image):
    corners = np.zeros((4, 1, 2), dtype=np.float32)
    corners = np.array([[[0,0]],[[0,image.shape[0]]],[[image.shape[1],0]],[[image.shape[1],image.shape[0]]]], dtype=np.float32)
    return corners


def getMatches(im1,im2):
   akDetect = cv2.KAZE_create()
   kp1, desc1=akDetect.detectAndCompute(im1,None)
   kp2, desc2=akDetect.detectAndCompute(im2,None)
   matcher = cv2.BFMatcher(cv2.NORM_L2)

   matches=matcher.knnMatch(desc2[:260000],trainDescriptors=desc1[:260000],k=2)
   return kp1, kp2, matches

def filter_matches(kp1, kp2, matches,  par, match_ratio=.75): 
   nkp1=[]
   nkp2=[]
   for i,m in enumerate(matches):
      x1=kp1[m[0].trainIdx].pt[0]
      x2=kp2[m[0].queryIdx].pt[0]
      y1=kp1[m[0].trainIdx].pt[1]
      y2=kp2[m[0].queryIdx].pt[1]
      dx=x2-x1
      dy=y2-y1
      #cond= (kp1[m[0].trainIdx].pt[0]>par["xmin"]) and (kp2[m[0].queryIdx].pt[0]>par["xmin"]) and (kp1[m[0].trainIdx].pt[0]<par["xmax"]) and (kp2[m[0].queryIdx].pt[0]<par["xmax"]) and ((kp1[m[0].trainIdx].pt[0]-kp2[m[0].queryIdx].pt[0])>par["dxmin"]) and (abs(kp1[m[0].trainIdx].pt[1]-kp2[m[0].queryIdx].pt[1])<par["dymax"]) and (abs(kp1[m[0].trainIdx].pt[1]>par["ymin"]) and (kp2[m[0].queryIdx].pt[1])>par["ymin"])
      cond= (x1>par["xmin1"]) and (x2>par["xmin2"]) and (x1<par["xmax1"]) and (x2<par["xmax2"]) and (np.abs(dx)>par["dxmin"]) and (np.abs(dx)<par["dxmax"]) and (np.abs(dy)>par["dymin"]) and (np.abs(dy)<par["dymax"])
      #cond=True
      if (len(m)==2 and m[0].distance<par["match_ratio"]*m[1].distance and cond):
         nkp1.append(kp1[m[0].trainIdx])  
         nkp2.append(kp2[m[0].queryIdx])
   p1=np.array([p.pt for p in nkp1])
   p2=np.array([p.pt for p in nkp2])
   return p1, p2

def draw_matches(im1, im2, p1, p2 ,imname, lwidth=5,r=8):
    cols = np.random.randint(0,256,[len(p1),3])*1.
    im = np.hstack((im1,im2))
    p2 = p2 + np.array([im1.shape[1], 0])
    p1=p1.astype(np.int)
    p2=p2.astype(np.int)
    for j in range(len(p1)):
       cv2.line(im, tuple(p1[j]), tuple(p2[j]), cols[j], lwidth)
       cv2.circle(im, tuple(p1[j]), r, cols[j], r)
       cv2.circle(im, tuple(p2[j]), r, cols[j], r)
    cv2.imwrite(imname,im)   

def blendImagePair(warped_image, image_1, image_2, point, homography):
    output_image = np.copy(warped_image)

    corners1 = getImageCorners(image_1)
    corners2 = getImageCorners(image_2)
    corners1Trans = cv2.perspectiveTransform(corners1,homography)

    mins = np.amin(corners1Trans, axis=0)
    maxs = np.amax(corners1Trans, axis=0)
    Img1_x_min = 0
    Img1_x_max = maxs[0][0] - mins[0][0]
    Img1_y_min = 0
    Img1_y_max = maxs[0][1] - mins[0][1]
    #print Img1_x_min," ",Img1_x_max," ",Img1_y_min," ",Img1_y_max

    maxs = np.amax(corners2, axis=0)
    Img2_x_min = int(point[0])
    Img2_x_max = int(maxs[0][0] + point[0])
    Img2_y_min = int(point[1])
    Img2_y_max = int(maxs[0][1] + point[1])
    #print "\n",Img2_x_min," ",Img2_x_max," ",Img2_y_min," ",Img2_y_max

    Int_x_min = max(Img1_x_min,Img2_x_min) #Int means Intersection
    Int_y_min = max(Img1_y_min,Img2_y_min)
    Int_x_max = min(Img1_x_max,Img2_x_max)
    Int_y_max = min(Img1_y_max,Img2_y_max)
    #print "\n",Int_x_min," ",Int_x_max," ",Int_y_min," ",Int_y_max
    #Compute Image
    output_image[point[1]:point[1] + image_2.shape[0],
                 point[0]:point[0] + image_2.shape[1]] = image_2
    #Handle Intersection Window
    Img1_x_centre = ( Img1_x_min + Img1_x_max ) / 2
    Img1_y_centre = ( Img1_y_min + Img1_y_max ) / 2
    max_distance1 =  int( ( Img1_x_max - Img1_x_centre ) +( Img1_y_max - Img1_y_centre ) )
    Img2_x_centre = ( Img2_x_min + Img2_x_max ) / 2
    Img2_y_centre = ( Img2_y_min + Img2_y_max ) / 2
    max_distance2 =  int( ( Img2_x_max - Img2_x_centre ) +( Img2_y_max - Img2_y_centre ) )

    for x in range(Int_x_min,Int_x_max):
        for y in range(Int_y_min,Int_y_max):
            distance2 = float( abs( x - Img2_x_centre) + abs( y - Img2_y_centre ) )
            alpha = distance2/max_distance2

            distance1 = float( abs( x - Img1_x_centre) + abs( y - Img1_y_centre ) )
            beta = distance1/max_distance1
            try:
                if not (warped_image[y,x].all() == 0 ):
                    if alpha > beta :
                        output_image[y,x] =   alpha * warped_image[y,x] +  ( 1-alpha ) * image_2[y-point[1],x-point[0]]
                    else :
                        output_image[y,x] =   (1-beta) * warped_image[y,x] +  beta * image_2[y-point[1],x-point[0]]
            except IndexError:
                continue

    return output_image

def warpImagePair(image_1, image_2, homography):

    corners1 = getImageCorners(image_1)
    corners2 = getImageCorners(image_2)

    #perspective transform
    corners1Trans = cv2.perspectiveTransform(corners1,homography)
    #corners1Trans = np.dot(homography,corners1)

    cornersAll=np.concatenate((corners1Trans,corners2))

    mins = np.amin(cornersAll, axis=0)
    maxs = np.amax(cornersAll, axis=0)
    x_min = mins[0][0]
    x_max = maxs[0][0]
    y_min = mins[0][1]
    y_max = maxs[0][1]

    translationM = [[1, 0, -1 * x_min],
                    [0, 1, -1 * y_min],
                    [0, 0, 1]]

    translatedHomography = np.dot(translationM, homography)

    warped_image = cv2.warpPerspective(image_1, translatedHomography, (x_max - x_min, y_max - y_min)) # apply transform to img2

    output_image = blendImagePair(warped_image, image_1, image_2,
                                  (-1 * x_min, -1 * y_min),homography)
    return output_image

def correct(im):
   fx=6142.4
   fy=6135.3
   cx=1248
   cy=1385
   K=np.array([[fx,0,  cx],
               [0, fy, cy],
               [0,  0,  1]])
   kr1 = 0.12
   kr2 = 0.21
   kr3 =0

   kt1=0.012
   kt2=-0.025

   dc=np.array([kr1,kr2,kt1,kt2])
   res=cv2.undistort(im, K, dc)
   return res

def estim(f1, f2, imdir, filter_params, resdir, plotit=False):
   t0=time.time()
   ima_0=cv2.imread(imdir+f1)
   imb_0=cv2.imread(imdir+f2)

   ima = cv2.cvtColor(ima_0, cv2.COLOR_BGR2GRAY)
   imb = cv2.cvtColor(imb_0, cv2.COLOR_BGR2GRAY)

   kp1,kp2,matches=getMatches(ima,imb)
   p1,p2=filter_matches(kp1, kp2, matches, filter_params) 
   print("%s matches found for %s and %s"%(len(p1), f1, f2))

   if len(p1):
      if plotit: draw_matches(ima_0, imb_0, p1, p2 , resdir+"/matches_%s_%s.png"%(f1,f2), lwidth=5,r=8)
      H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
      HR = cv2.estimateAffinePartial2D(p1, p2)
      np.save(resdir+"/H_%s_%s.npy"%(f1,f2),H)
      np.save(resdir+"/H_%s_%s_p1.npy"%(f1,f2),p1)
      np.save(resdir+"/H_%s_%s_p2.npy"%(f1,f2),p2)
      np.save(resdir+"/HR_%s_%s.npy"%(f1,f2),HR)
   else: 
      H=None
      HR=None   
   print("H=", H)
   t1=time.time()
   print("it took", t1-t0, "seconds")