import numpy as np
import cv2
import glob
import os

#example image
folder_l = 'images_left'
folder_r = 'images_right'
images = [os.path.basename(x) for x in glob.glob('./'+str(folder_l)+'/*.png')]
img_l = cv2.imread('./'+str(folder_l)+'/'+str(images[0]))
img_r = cv2.imread('./'+str(folder_r)+'/'+str(images[0]))
h,  w = img_l.shape[:2]

img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

#img_l = cv2.cvtColor(img_l,cv2.COLOR_BGR2GRAY)
#img_r = cv2.cvtColor(img_r,cv2.COLOR_BGR2GRAY)
'''
imgpoints_l = np.load('./'+str(folder_l)+'/imgpoints.npy')
imgpoints_r = np.load('./'+str(folder_r)+'/imgpoints.npy')
filenames_l = np.load('./'+str(folder_l)+'/filenames.npy')
filenames_r = np.load('./'+str(folder_r)+'/filenames.npy')

n_corners = (9,6)
objp = np.zeros((n_corners[0]*n_corners[1],3), np.float32)
objp[:,:2] = np.mgrid[0:n_corners[0],0:n_corners[1]].T.reshape(-1,2)
objpoints = [objp]*len(filenames_l)
'''

mapx_l = np.load('./'+str(folder_l)+'/mapx.npy')
mapy_l = np.load('./'+str(folder_l)+'/mapy.npy')
img_l = cv2.remap(img_l,mapx_l,mapy_l,cv2.INTER_CUBIC)

mapx_r = np.load('./'+str(folder_r)+'/mapx.npy')
mapy_r = np.load('./'+str(folder_r)+'/mapy.npy')
img_r = cv2.remap(img_r,mapx_r,mapy_r,cv2.INTER_CUBIC)

# replace with example image
#img_l = cv2.imread('./example/tsukuba_l.png')
#img_r = cv2.imread('./example/tsukuba_r.png')
#img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
#img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

#compute disparity v1
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(img_l,img_r)
'''
#compute disparity v2 (sloooow)
from CostFilter import computeDisp
max_disp = 15
scale_factor = 16
disparity = computeDisp(img_l, img_r, max_disp) * scale_factor
'''
disparity = (disparity-disparity.min())/(disparity.max()-disparity.min())
cv2.imshow( 'disparity', disparity) #display image
cv2.moveWindow('disparity', 50, 50) #window position

key = cv2.waitKey(0)
cv2.destroyAllWindows()