import os
import cv2
import numpy as np
import time

DEBUGGING = True
DRAWMATCHES = True
PRINTDETAILS = False


def pts2D_TO_pts3D(pts1_2D, pts2_2D):
    """
    Convert 2D points (u,v) to 3D points (u, v, 1)\n
    Params:
    -------
        pts1_2D - pts1 2D points\n
        pts2_2D - pts1 2D points\n
    Retuns:
    -------
        pts1_3D, pts2_3D points
    """
    pts1_3D = np.ones((pts1_2D.shape[0], 3), np.float32)
    pts2_3D = np.ones((pts2_2D.shape[0], 3), np.float32)

    for d1_2D, d2_2D, p1_3D, p2_3D in zip(pts1_2D, pts2_2D, pts1_3D, pts2_3D):

        p1_3D[0], p1_3D[1] = d1_2D[0], d1_2D[1]
        p2_3D[0], p2_3D[1] = d2_2D[0], d2_2D[1]

        if PRINTDETAILS:
            print(f'p1_2D: {d1_2D}      p2_2D: {d2_2D}')
            print(f'p1_3D: {p1_3D} p2_3D: {p2_3D}')

    return pts1_3D, pts2_3D

def test_epipolar(F, i, p1, p2):
    """
    Test if the epipolar lines are horizonontal\n
    Params:
    -------
        F: Fundamental matrix\n
        i: iteration value\n
        p1: points left camera\n
        p2: points right camera\n
    """
    # test epipolar
    # TODO check this equation
    # raw = abs(p2.T @ F @ p1)
    raw = abs(p2.T.dot(F).dot(p1))
    is_horizontal = np.round(raw)

    if PRINTDETAILS:
        print('raw value ', raw)
        print(f'p2_{i}.T @ F @ p1_{i} = ', is_horizontal)

    if is_horizontal != 0: 
        print('fond no horizontal epipolar line')
        return False

    return True

def drawlines(img1,img2,lines,pts1,pts2):
    """
    img1 - image on which we draw the epilines for the points in img2
    lines - corresponding epilines
    """

    if PRINTDETAILS:
       print (f' -- img1.shape: {img1.shape} img2.shape:{img2.shape}')

    r,c, = img1.shape[:2]

    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c) / r[1]])
        cv2.line(img1, (x0, y0), (x1, y1), color,1)
        cv2.circle(img1, (int(pt1[0]), int(pt1[1])), 5, color, -1)
        cv2.circle(img2, (int(pt2[0]), int(pt2[1])), 5, color, -1)

    return img1, img2

def stereoRectifyUncalibrated(imgPoints1, imgPoints2, F, w, h):
    """
    Stereo vision rectification in the map.\n
    1. Stereo rectify uncalibrated camera.\n
    2. Get optimal new camera matrix for the left and the right camera.\n
            - getOptimalNewCameraMatrix
    3. Init undistort rectify map for the left and right camera.\n
            - initUndistortRectifyMap
    """
    # 1.
    ( retval, H1, H2 )  = cv2.stereoRectifyUncalibrated (
                                    points1 = imgPoints1,
                                    points2 = imgPoints2,
                                    F = F,
                                    imgSize = (w, h),
                                    threshold = 5
                                        )
    print(f' -- image recitfication done - {retval}')

    return retval, H1, H2

def compute_keypoints_and_rectify(imgL, imgR, ratio_threshold, corner_threshold):
        
        cv2.swift = cv2.SIFT_create()

        # Convert to CV_8UC1
        grayL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)

        #get points in frame L
        #get sift points to match (kp1)
        kp1, des1 = cv2.swift.detectAndCompute(grayL, None)

        #get points in frame R
        #get sift points to match (kp2)
        kp2, des2 = cv2.swift.detectAndCompute(grayR, None)

        #only take points that are in both images to compute fundamental matrix
        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary

        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)

        good_matches = []
        pts1_2D = []
        pts2_2D = []
        # ratio test as per Lowe's paper
        for m,n in matches:

            # make sure the distance to the closest match is sufficiently better than the second closest
            if (m.distance < ratio_threshold * n.distance and
                kp1[m.queryIdx].response > corner_threshold and
                kp2[m.trainIdx].response > corner_threshold):

                good_matches.append((m.queryIdx, m.trainIdx))
                pts2_2D.append(kp2[m.trainIdx].pt)
                pts1_2D.append(kp1[m.queryIdx].pt)
        
        pts1_2D = np.int32(pts1_2D)
        pts2_2D = np.int32(pts2_2D)
        print(f'    treshold ratio: {ratio_threshold}')
        print(f'   corner treshold: {corner_threshold}')
        print(f'      good machtes: {len(good_matches)}')

        if pts1_2D.size == 0:
            return imgL, imgR
        
        if pts2_2D.size == 0:
            return imgL, imgR

        #compute fundamental matrix F based on given keypoints (F)
        F, mask = cv2.findFundamentalMat(pts1_2D.astype(float),pts2_2D.astype(float), cv2.FM_RANSAC)# cv2.FM_LMEDS)

        if F is None:
            print('  ----<>---- no points found')
            return
        
        # flip the fundamental matrix
        if np.linalg.det(F) < 0 :
            F = -F

        # remove points where the mask is equals 0
        # select only inliers points
        pts1_2D = pts1_2D[mask.ravel() == 1]
        pts2_2D = pts2_2D[mask.ravel() == 1]

        pts1_2D = np.float32(pts1_2D)
        pts2_2D = np.float32(pts2_2D)

        if DEBUGGING:

            # validate points
            assert pts1_2D.shape[0] == pts2_2D.shape[0], "number of points don't match"

            pts1_3D, pts2_3D = pts2D_TO_pts3D(pts1_2D=pts1_2D, pts2_2D=pts2_2D)

            print(f' -- p1_2D: {len(pts1_2D)}')
            print(f' -- p2_2D: {len(pts2_2D)}')

            # optimize this loop with np.tile
            for i in range(len(pts1_3D)):
                # validate the fundamental matrix equation
                p1 = pts1_3D.T[:, i]
                p2 = pts2_3D.T[:, i]
                ret_val = test_epipolar(F=F, i=i, p1=p1, p2=p2)
                if not ret_val:
                    return False, None, None 

            # Find epilines corresponding to points in right image (second image) and
            # drawing its lines on left image
            lines1 = cv2.computeCorrespondEpilines(pts2_2D.reshape(-1,1,2), 2, F)
            lines1 = lines1.reshape(-1,3)

            # Find epilines corresponding to points in left image (first image) and
            # drawing its lines on right image
            lines2 = cv2.computeCorrespondEpilines(pts1_2D.reshape(-1,1,2), 1, F)
            lines2 = lines2.reshape(-1,3)

            if DRAWMATCHES:
                img5,img6 = drawlines(imgL,imgR,lines1,pts1_2D,pts2_2D)
                img3,img4 = drawlines(imgR,imgL,lines2,pts2_2D,pts1_2D)

                img5 = cv2.resize(img5, (300,300))
                img3 = cv2.resize(img3, (300,300))

                hstack = np.hstack([img5, img3])
                cv2.imshow('input left-right images (300,300)', hstack)

        ######### stereo rectification region ########################
        # need to reshape vector so that stereorectifyuncalibrated can
        # convert pts Nx2 to pts_n Nx1 
        pts1_n_1D = pts1_2D.reshape((pts1_2D.shape[0] * 2, 1))
        pts2_n_1D = pts2_2D.reshape((pts2_2D.shape[0] * 2, 1))

        #compute rectification transform
        imgPoints1 = pts1_n_1D
        imgPoints2= pts2_n_1D

        h, w = imgL.shape[:2]
        # calculate the homography matrix
        _, H1, H2 = stereoRectifyUncalibrated(imgPoints1, imgPoints2, F, w, h)

        #apply rectification matrix
        h, w = imgL.shape[:2]
        # h = int(h * 2)
        # w = int(w * 2)
        rec_imgL = cv2.warpPerspective(
                        src=imgL, 
                        M=H1, 
                        dsize=(w,h), 
                        flags=cv2.WARP_FILL_OUTLIERS
                        )
        rec_imgR = cv2.warpPerspective(
                        src=imgR, 
                        M=H2,
                        dsize=(w,h), 
                        flags=cv2.WARP_FILL_OUTLIERS
                        )

        return True, rec_imgL, rec_imgR


if __name__ =='__main__':

    # print(__file__)
    # print(os.path.dirname(__file__))

    iL = os.path.realpath(os.path.join(os.path.dirname(__file__), 'imgs/cropped/', '0_ROI_1369x1330_CAM_0.bmp'))
    iR = os.path.realpath(os.path.join(os.path.dirname(__file__), 'imgs/cropped/', '0_ROI_1369x1330_CAM_1.bmp'))

    imgL = cv2.imread(iL)
    imgR = cv2.imread(iR)

    thresh = 70
    for i in range(30):

        imL = imgL.copy()
        imR = imgR.copy()

        t = thresh/100
        ret_val, imL_rec, imR_rec = compute_keypoints_and_rectify(imgL=imL, imgR=imR, ratio_threshold=t, corner_threshold=0)

        print('thresh: ', thresh)
        thresh = thresh + 1
        if not ret_val:
            continue

        imL_rec = cv2.resize(imL_rec, (300,300))
        imR_rec = cv2.resize(imR_rec, (300,300))
        hstack = np.hstack([imL_rec, imR_rec])
        cv2.imshow('rectified left-right images (300,300)', hstack)
        cv2.waitKey(1)
        time.sleep(1)


    cv2.waitKey(0)
    cv2.destroyAllWindows()



