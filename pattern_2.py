# 2017-11-08 in Tokyo by H.J.T.
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html

# coding: utf-8
import numpy as np
import cv2
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10
#img1 = cv2.imread('/Users/huangjintao/Desktop/pattern/Test.jpg',0) # queryImage
img1 = cv2.imread('/Users/huangjintao/Desktop/pattern/20171015wf105D3200DSC_0420s.JPG') # queryImage
img2 = cv2.imread('/Users/huangjintao/Desktop/pattern/20171015sw105D3200DSC_0420s.JPG') # trainImage
#img2 = cv2.imread('/Users/huangjintao/Desktop/pattern/vesamwf.jpg')  # trainImage

# Initiate SIFT detector
#sift = cv2.SIFT()
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w,_= img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

#================================================================

    print([np.int32(dst)]) # img2-ROI area
    xx = np.int32(dst)[0][0][0]
    ww = np.int32(dst)[3][0][0]
    yy = np.int32(dst)[0][0][1]
    hh = np.int32(dst)[1][0][1]
    print(xx,ww,yy,hh)
    img_ROI = img2[yy:hh,xx:ww]
    cv2.imwrite('/Users/huangjintao/Desktop/pattern/img_ROI.png', img_ROI)
    img_tran = img2

    #hhh, www, _ = img_ROI.shape
    #print(www, hhh)
    logo = cv2.resize(img1, (ww-xx, hh-yy), interpolation=cv2.INTER_AREA)
    print(ww-xx, hh-yy)

    img2gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    roi = img_tran[yy:hh,xx:ww]

    # black-out the area of queryImage in ROI
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    # Take only region of queryImage from image.
    img2_fg = cv2.bitwise_and(logo, logo, mask=mask)
    # Put lqueryImage in ROI and modify the main image
    result = cv2.add(img1_bg, img2_fg)
    img_tran[yy:hh,xx:ww] = result

    cv2.imwrite('/Users/huangjintao/Desktop/pattern/result.png', img_tran)
#================================================================

    # 画出ROI区域
    img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

else:
    print ("Not enough matches are found")
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

plt.imshow(img3),plt.show()