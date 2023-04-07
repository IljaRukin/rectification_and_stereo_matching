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

img_l = cv2.cvtColor(img_l,cv2.COLOR_BGR2GRAY)
img_r = cv2.cvtColor(img_r,cv2.COLOR_BGR2GRAY)

imgpoints_l = np.load('./'+str(folder_l)+'/imgpoints.npy')
imgpoints_r = np.load('./'+str(folder_r)+'/imgpoints.npy')
filenames_l = np.load('./'+str(folder_l)+'/filenames.npy')
filenames_r = np.load('./'+str(folder_r)+'/filenames.npy')

n_corners = (9,6)
objp = np.zeros((n_corners[0]*n_corners[1],3), np.float32)
objp[:,:2] = np.mgrid[0:n_corners[0],0:n_corners[1]].T.reshape(-1,2)
objpoints = [objp]*len(filenames_l)

#calibrate
#ret = average mean square error in pixel
#mtx = camera matrix
#dist = distorsion
#rvecs/tvecs = rotation/translation vectors
ret, mtx_l, dist_l, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints_l, (w,h),(w,h),None)
ret, mtx_r, dist_r, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints_r, (w,h),(w,h),None)

#calculate other parameters (mtx_l, dist_l, mtx_r, dist_r stay the same)
#R = rotation matrix
#T = translation matrix
#E = essential matrix
#F = fundamental matrix
ret, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints_l, imgpoints_r, mtx_l, dist_l,mtx_r, dist_r, (w,h))

#more calculation
#R_l = rotation matrix for left camera
#R_r – rotation matrix for right camera
#P_l = 3x4 projection matrix into rectified coordinates for left camera
#P_r = 3x4 projection matrix into rectified coordinates for right camera
#Q = 4x4 disparity-to-depth mapping matrix
#roi = bounds for croping image
#############(see reprojectImageTo3D() ).
R_l, R_r, P_l, P_r, Q, roi_l, roi_r = cv2.stereoRectify(mtx_l, dist_l, mtx_r, dist_r, (w,h), R, T)

#undistort
#mapx_l,mapy_l = cv2.initUndistortRectifyMap(mtx_l,dist_l,R_l,P_l,(w,h),cv2.CV_32FC1)
#mapx_r,mapy_r = cv2.initUndistortRectifyMap(mtx_r,dist_r,R_r,P_r,(w,h),cv2.CV_32FC1)
#dst_l = cv2.remap(img_l,mapx_l,mapy_l,cv2.INTER_CUBIC)
#dst_r = cv2.remap(img_r,mapx_r,mapy_r,cv2.INTER_CUBIC)

# crop the image
#x,y,w,h = roi_l
#dst_l = dst_l[y:y+h, x:x+w]
#x,y,w,h = roi_r
#dst_r = dst_r[y:y+h, x:x+w]
'''
cv2.imshow( 'original image', cv2.cvtColor(img_l,cv2.COLOR_BGR2RGB)) #display image
cv2.moveWindow('original image', 50, 50) #window position
cv2.imshow( 'undistorted image', cv2.cvtColor(dst_l,cv2.COLOR_BGR2RGB)) #display image
cv2.moveWindow('undistorted image', 550, 50) #window position
cv2.imshow( 'original image', cv2.cvtColor(img_r,cv2.COLOR_BGR2RGB)) #display image
cv2.moveWindow('original image', 50, 50) #window position
cv2.imshow( 'undistorted image', cv2.cvtColor(dst_r,cv2.COLOR_BGR2RGB)) #display image
cv2.moveWindow('undistorted image', 550, 50) #window position
'''

'''
# find flann features
sift = cv2.SIFT_create()
# find the keypoints and descriptors with SIFT
kp_l, des_l = sift.detectAndCompute(img_l,None)
kp_r, des_r = sift.detectAndCompute(img_r,None)
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des_l,des_r,k=2)
pts_l = []
pts_r = []
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        pts_r.append(kp_r[m.trainIdx].pt)
        pts_l.append(kp_l[m.queryIdx].pt)

# remove outer points
pts_l = np.int32(pts_l)
pts_r = np.int32(pts_r)
F, mask = cv2.findFundamentalMat(pts_l,pts_r,cv2.FM_LMEDS)
# We select only inlier points
pts_l = pts_l[mask.ravel()==1]
pts_r = pts_r[mask.ravel()==1]
'''
pts_l = np.round(imgpoints_l[0]).astype(int)
pts_r = np.round(imgpoints_r[0]).astype(int)
pts_l = np.reshape(pts_l,[pts_l.shape[0],pts_l.shape[2]])
pts_r = np.reshape(pts_r,[pts_r.shape[0],pts_r.shape[2]])

print(pts_l.shape)
# function for drawing epipolar lines
def drawlines(img_l,img_r,lines,pts_l,pts_r):
    ''' img_l - image on which we draw the epilines for the points in img_r
        lines - corresponding epilines '''
    r,c = img_l.shape[0:2]
    img_l = cv2.cvtColor(img_l,cv2.COLOR_GRAY2BGR)
    img_r = cv2.cvtColor(img_r,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts_l,pts_r):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img_l = cv2.line(img_l, (x0,y0), (x1,y1), color,1)
        img_l = cv2.circle(img_l,tuple(pt1),5,color,-1)
        img_r = cv2.circle(img_r,tuple(pt2),5,color,-1)
    return img_l,img_r

# find epipolar lines corresponding to points in right image (second image) and
# drawing its lines on left image
lines_r = cv2.computeCorrespondEpilines(pts_r.reshape(-1,1,2), 2,F)
lines_r = lines_r.reshape(-1,3)
img_epi_l,img_dot_r = drawlines(img_l,img_r,lines_r,pts_l,pts_r)
# find epipolar lines corresponding to points in left image (first image) and
# drawing its lines on right image
lines_l = cv2.computeCorrespondEpilines(pts_l.reshape(-1,1,2), 1,F)
lines_l = lines_l.reshape(-1,3)
img_epi_r,img_dot_l = drawlines(img_r,img_l,lines_l,pts_r,pts_l)

cv2.imshow( 'epipolar lines left', cv2.cvtColor(img_epi_l,cv2.COLOR_BGR2RGB)) #display image
cv2.moveWindow('epipolar lines left', 50, 50) #window position
cv2.imshow( 'epipolar lines right', cv2.cvtColor(img_epi_r,cv2.COLOR_BGR2RGB)) #display image
cv2.moveWindow('epipolar lines right', 550, 50) #window position

key = cv2.waitKey(0)
cv2.destroyAllWindows()