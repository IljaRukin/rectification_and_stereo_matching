import numpy as np
import cv2
import glob
import os
import sys
from time import sleep

# termination criteria
# 10 iterations, 0.01 accuracy
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

n_corners = (9,6)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((n_corners[0]*n_corners[1],3), np.float32)
objp[:,:2] = np.mgrid[0:n_corners[0],0:n_corners[1]].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
imgpoints_l = [] # 2d points in image plane.
imgpoints_r = [] # 2d points in image plane.
filenames_l = [] # filenames of included images
filenames_r = [] # filenames of included images

folder_l = 'images_left'
folder_r = 'images_right'

images = [os.path.basename(x) for x in glob.glob('./'+str(folder_l)+'/*.png')]
#images_r = [os.path.basename(x) for x in glob.glob('./'+str(folder_r)+'/*.png')]
for fname in images:
    img_l = cv2.imread('./'+str(folder_l)+'/'+str(fname))
    img_r = cv2.imread('./'+str(folder_r)+'/'+str(fname))
    print(str(fname))
    gray_l = cv2.cvtColor(img_l,cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret_l, corners_l = cv2.findChessboardCorners(gray_l, n_corners,None,cv2.CALIB_CB_FAST_CHECK)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, n_corners,None,cv2.CALIB_CB_FAST_CHECK)

    # If found, refine and display image points
    if (ret_l == True) and (ret_r == True) :
        corners2_l = cv2.cornerSubPix(gray_l,corners_l,(11,11),(-1,-1),criteria)
        corners2_r = cv2.cornerSubPix(gray_r,corners_r,(11,11),(-1,-1),criteria)

        # Draw and display the corners
        scaler = 1
        res_l = tuple( [int(k*scaler) for k in img_l.shape[1::-1]] )
        img_l = cv2.drawChessboardCorners(cv2.resize( img_l , res_l), n_corners, np.array(corners2_l)*scaler,ret_l) #resize
        cv2.imshow(fname, img_l) #display image
        cv2.moveWindow(fname, 50, 50) #window position

        #if user approves, add object points
        key1 = cv2.waitKey(0)

        if key1 == ord('y'):
            #print('append image')
            pass
        elif key1 == ord('n'):
            #print('discard image')
            continue
        else:
            #print('quit')
            sys.exit()

        # Draw and display the corners
        scaler = 1
        res_r = tuple( [int(k*scaler) for k in img_r.shape[1::-1]] )
        img_r = cv2.drawChessboardCorners(cv2.resize( img_r , res_r), n_corners, np.array(corners2_r)*scaler,ret_r) #resize
        cv2.imshow(fname, img_r) #display image
        cv2.moveWindow(fname, 50, 50) #window position

        #if user approves, add object points
        key2 = cv2.waitKey(0)

        if (key1 == ord('y')) and (key2 == ord('y')):
            #print('append image')
            imgpoints_l.append(corners2_l)
            imgpoints_r.append(corners2_r)
            filenames_l.append('./'+str(folder_l)+'/'+str(fname))
            filenames_r.append('./'+str(folder_r)+'/'+str(fname))
        elif key2 == ord('n'):
            #print('discard image')
            continue
        else:
            #print('quit')
            sys.exit()

        cv2.destroyAllWindows()

np.save('./'+str(folder_l)+'/imgpoints.npy',imgpoints_l)
np.save('./'+str(folder_r)+'/imgpoints.npy',imgpoints_r)
np.save('./'+str(folder_l)+'/filenames.npy',filenames_l)
np.save('./'+str(folder_r)+'/filenames.npy',filenames_r)

cv2.destroyAllWindows()