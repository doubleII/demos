import sys

# import the local packages
sys.path.append("/Developement/doubleii-stereo-vision/tools")

import os
import numpy as np
import cv2
import logging
import copy

from pckgs.imgfilter import by3x3d9_kernel, by5x5d25_kernel
from pckgs.constants import PATH_TO_IMGS
from pckgs.images_from import from_dir

from modules.rectification import undistort
from modules.utils1 import stack_images, create_black_frame

TESTING = True


class Measure(object):
    """ """

    def __init__(self, trackbar=False, win_name="MeasureTresholdTrackBars"):
        self.trackbar = trackbar
        self.win_name = win_name
        self.black_frame = None
        if self.trackbar:
            self.init_trackbars()

    def empty(self, a):
        pass

    def init_trackbars(self):
        """init the trackbars. need to run only once"""
        cv2.namedWindow(winname=self.win_name)
        cv2.resizeWindow(self.win_name, 840, 470)
        cv2.createTrackbar("Min", self.win_name, 10, 255, self.empty)
        cv2.createTrackbar("Max", self.win_name, 110, 255, self.empty)
        # test
        cv2.createTrackbar("Wide Min", self.win_name, 10, 255, self.empty)
        cv2.createTrackbar("Wide Max", self.win_name, 200, 255, self.empty)
        cv2.createTrackbar("Mid Min", self.win_name, 30, 255, self.empty)
        cv2.createTrackbar("Mid Max", self.win_name, 150, 255, self.empty)
        cv2.createTrackbar("Tight Min", self.win_name, 240, 255, self.empty)
        cv2.createTrackbar("Tight Max", self.win_name, 250, 255, self.empty)
        cv2.createTrackbar("Dilatation", self.win_name, 2, 20, self.empty)
        cv2.createTrackbar("Erosion", self.win_name, 1, 20, self.empty)

    def get_trackbar_values(self):
        """get the trackbar values in runtime

        `Vals: thresh, thr_wide, thr_mid, thr_tight`
        """
        # background
        thresh1 = cv2.getTrackbarPos("Min", self.win_name)
        # object
        thresh2 = cv2.getTrackbarPos("Max", self.win_name)
        # test thresholds
        thr_wide1 = cv2.getTrackbarPos("Wide Min", self.win_name)
        thr_wide2 = cv2.getTrackbarPos("Wide Max", self.win_name)
        thr_mid1 = cv2.getTrackbarPos("Mid Min", self.win_name)
        thr_mid2 = cv2.getTrackbarPos("Mid Max", self.win_name)
        thr_tight1 = cv2.getTrackbarPos("Tight Min", self.win_name)
        thr_tight2 = cv2.getTrackbarPos("Tight Max", self.win_name)
        dilatation = cv2.getTrackbarPos("Dilatation", self.win_name)
        erosion = cv2.getTrackbarPos("Erosion", self.win_name)

        vals = {
            "thresh": [thresh1, thresh2],
            "thr_wide": [thr_wide1, thr_wide2],
            "thr_mid": [thr_mid1, thr_mid2],
            "thr_tight": [thr_tight1, thr_tight2],
            "dilatation": [dilatation],
            "erosion": [erosion],
        }

        print(vals)
        return vals

    def reorder(self, points):
        """
        reorder the shape
        1. calc min and max (0,0) (w,h)
        2. calc diff in rows axis=1 means rows and find (h,0) (w, 0)
        P1(0,0)-------(w,0) P2
              |       |
              |       |
              |       |
        P3(0,h)-------(w,h) P4
        """
        print("reordered points:\n", points)
        # shape should be recktangle
        shape = (4, 1, 2)
        if points.shape != shape:
            # for exact measurement the shape should be rectangle or quader
            print(f"\033[35m reorder: shape should be rectangle {points.shape}.\033[0m")
            return None, False

        pts_new = np.zeros_like(points)
        points = points.reshape((4, 2))
        add = points.sum(1)
        pts_new[0] = points[np.argmin(add)]
        pts_new[3] = points[np.argmax(add)]
        diff = np.diff(points, axis=1)
        pts_new[1] = points[np.argmin(diff)]
        pts_new[2] = points[np.argmax(diff)]

        return pts_new, True

    def warp_img(self, img, points, w, h):
        """

        Returns:
        -------
        Ok: warped image, True
        Not Ok: img, False
        """
        try:
            # reorder the points
            points, success = self.reorder(points=points)

            if not success:
                return img, False

            pts1 = np.float32(points)
            # background
            pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            warp_img = cv2.warpPerspective(src=img, M=matrix, dsize=(w, h))
            warp_img = self.padding(img=warp_img)
            return warp_img, True

        except Exception as err:
            print(f"\033[31m warp_img: {err}\033[0m")
            return img, False

    def get_background(self, contoures):
        """get the background working area if contoures is not empty

        Returns:
        -------
        Ok: contours, True
        Not Ok: None, False
        """

        if contoures is None:
            print("\033[31m No background found\033[0m")
            return None, False
        if len(contoures) <= 0:
            print("\033[31m No background found\033[0m")
            return None, False

        return contoures[0]["approx"], True

    def get_contours(
        self,
        img,
        min_area=0,
        filter=0,
        show_canny=False,
        size=[612, 512],
        draw=False,
        wait_key=1,
        background=False,
    ):
        """
        convert the image th CV_8UC1, GaussianBlur, Canny, dilation, erode
        find the background working area.

        Params:
        -------
        img, min_area (default -1, by background should be a larger),
        filter by default 0 for unknown object detection. The background has 4 corners (or for obj. with more as 4 corners),
        rectangle has 4 corners. canny_show by default false (show filtered image),
        size by default [612,512], draw (contours) by default false, wait_key 0 or 1

        Returns:
        --------
        Ok: img, final_corners, True
        Not Ok: None, None, False
        """
        try:
            vals = self.get_trackbar_values()
            c_thr = vals["thresh"]
            if len(img.shape) > 2:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            # blurred = cv2.GaussianBlur(src=gray, ksize=(5,5), sigmaX=0, sigmaY=0)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            canny = cv2.Canny(blurred, threshold1=c_thr[0], threshold2=c_thr[1])
            k = by3x3d9_kernel()
            dilation = cv2.dilate(
                src=canny, kernel=k, iterations=int(vals["dilatation"][0])
            )
            erosion = cv2.erode(
                src=dilation, kernel=k, iterations=int(vals["erosion"][0])
            )
            thresh = erosion

            if show_canny:
                imm = stack_images(scale=0.15, imgArray=[[canny, dilation, thresh]])
                cv2.imshow("canny-dilation-threshold", imm)
                cv2.waitKey(wait_key)

            contoures, hierarchy = cv2.findContours(
                image=thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE
            )
            final_contours = []
            for cnt in contoures:
                area = cv2.contourArea(cnt)
                if area > min_area:
                    perimeter = cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
                    # bbox = x,y,w,h
                    x, y, w, h = cv2.boundingRect(array=approx)
                    cx, cy = x + (w // 2), y + (h // 2)

                    # contours with filter
                    if filter > 0:
                        if len(approx) == filter:
                            final_contours.append(
                                {
                                    "corner": len(approx),
                                    "area": area,
                                    "approx": approx,
                                    "bbox": [x, y, w, h],
                                    "cnt": cnt,
                                    "center": [cx, cy],
                                }
                            )
                    # conours without filter (filter=0)
                    else:
                        final_contours.append(
                            {
                                "corner": len(approx),
                                "area": area,
                                "approx": approx,
                                "bbox": [x, y, w, h],
                                "cnt": cnt,
                                "center": [cx, cy],
                            }
                        )
            # in descending the biggers shape (area) first
            final_contours = sorted(
                final_contours, key=lambda x: x["area"], reverse=True
            )

            if draw:
                for con in final_contours:
                    # draw contours
                    img = cv2.drawContours(img, con["cnt"], -1, (255, 0, 255), 3)

                    xx, yy, ww, hh = cv2.boundingRect(con["approx"])
                    # used for shapes not for background
                    if not background:
                        corners = con["corner"]
                        if corners == 3:
                            objectType = "Tri"
                        elif corners == 4:
                            aspRatio = ww / float(hh)
                            if aspRatio > 0.95 and aspRatio < 1.05:
                                objectType = "Square"
                            else:
                                objectType = "Rectangle"
                        elif corners > 4:
                            objectType = "Circles"
                        else:
                            objectType = "None"

                        cv2.putText(
                            img=img,
                            text=objectType,
                            org=(xx + (ww // 2) - 10, yy + (hh // 2) - 10),
                            fontFace=cv2.FONT_HERSHEY_COMPLEX,
                            fontScale=1,
                            color=(0, 0, 0),
                            thickness=1,
                        )
                        # draw the center point
                        cv2.circle(
                            img=img,
                            center=(xx + (ww // 2), yy + (hh // 2)),
                            radius=3,
                            color=(0, 0, 255),
                            thickness=-1,
                        )

                    img = cv2.rectangle(
                        img, (xx, yy), (xx + ww, yy + hh), (0, 255, 0), 2
                    )

            return img, final_contours, True

        except Exception as ex:
            print(f"\033[31m get_contours: {ex}\033[0m")
            return self.black_frame, None, False

    def padding(self, img, pad=10):
        return img[pad : img.shape[0] - pad, pad : img.shape[1] - pad]

    def compute_simple_shape(self, img, contours, scale, text=True):
        """compute object contours

        Returns:
        -------
        Ok: img, True
        Not Ok: img, False
        """
        try:
            if len(contours) != 0:
                for cnt in contours:
                    new_pts, _ = self.reorder(cnt[2])

                    calc_w = round(
                        self.calc_distance(
                            pts1=new_pts[0][0] // scale, pts2=new_pts[1][0] // scale
                        ),
                        5,
                    )
                    calc_h = round(
                        self.calc_distance(
                            pts1=new_pts[0][0] // scale, pts2=new_pts[2][0] // scale
                        ),
                        5,
                    )

                    if text:
                        img = cv2.putText(
                            img=img,
                            text=f"units: mm",
                            org=(10, 15),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            color=(0, 255, 0),
                            thickness=1,
                            lineType=cv2.LINE_AA,
                        )

                        img = cv2.putText(
                            img=img,
                            text=f"w:{calc_w}",
                            org=(10, 30),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            color=(0, 255, 0),
                            thickness=1,
                            lineType=cv2.LINE_AA,
                        )

                        img = cv2.putText(
                            img=img,
                            text=f"pnts:{cnt[0]}",
                            org=(10, 45),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            color=(0, 255, 0),
                            thickness=1,
                            lineType=cv2.LINE_AA,
                        )

                        img = cv2.putText(
                            img=img,
                            text=f"h:{calc_h}",
                            org=(10, 60),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            color=(0, 255, 0),
                            thickness=1,
                            lineType=cv2.LINE_AA,
                        )

                    # !!! break because i get only the biggers shape !!!
                    print(" -- !!! break because i get only the biggers shape !!!")
                    break
            return img, True
        except Exception as ex:
            print(f"\033[31m compute_simple_shape: {ex} \033[0m")
            return img, False

    # now not used. same as compute_simle_shape
    def draw_simple_shape(self, img, contours, scale):
        """draw simple object contours

        Returns:
        -------
        Ok: img, True
        Not Ok: img, False
        """
        try:
            if len(contours) != 0:
                for cnt in contours:
                    # cnt[2] - approx
                    cv2.polylines(
                        img=img,
                        pts=[cnt[2]],
                        isClosed=True,
                        color=(0, 255, 0),
                        thickness=2,
                    )

                    new_pts, _ = self.reorder(cnt[2])

                    calc_w = round(
                        self.calc_distance(
                            pts1=new_pts[0][0] // scale, pts2=new_pts[1][0] // scale
                        ),
                        5,
                    )
                    calc_h = round(
                        self.calc_distance(
                            pts1=new_pts[0][0] // scale, pts2=new_pts[2][0] // scale
                        ),
                        5,
                    )

                    img = cv2.putText(
                        img=img,
                        text=f"w={calc_w} mm, Area={cnt[0]}",
                        org=(10, 30),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=(0, 255, 0),
                        thickness=1,
                        lineType=cv2.LINE_AA,
                    )

                    img = cv2.putText(
                        img=img,
                        text=f"h={calc_h} mm",
                        org=(10, 60),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=(0, 255, 0),
                        thickness=1,
                        lineType=cv2.LINE_AA,
                    )

                    # !!! break because i get only the biggers shape !!!
                    print(" -- !!! break because i get only the biggers shape !!!")
                    break
            return img, True
        except Exception as ex:
            print(f"\033[31m draw_simple_shape: {ex} \033[0m")
            return img, False

    def calc_distance(self, pts1, pts2):
        return np.sqrt((pts2[0] - pts1[0]) ** 2 + (pts2[1] - pts1[1]) ** 2)


    def get_contours_test(self, img):
        im = copy.deepcopy(img)
        vals = self.get_trackbar_values()
        thr_wide = vals["thr_wide"]
        thr_mid = vals["thr_mid"]
        thr_tight = vals["thr_tight"]
        """convert the image th CV_8UC1, GaussianBlur, Canny"""
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        wide = cv2.Canny(blurred, thr_wide[0], thr_wide[1])
        mid = cv2.Canny(blurred, thr_mid[0], thr_mid[1])
        tight = cv2.Canny(blurred, thr_tight[0], thr_tight[1])

        imgs = stack_images(scale=0.25, imgArray=[[wide, mid, tight]])
        cv2.imshow("wide-mid-tigh threshold", imgs)


if TESTING:
    if __name__ == "__main__":
        ######## settings with background paper size ####################
        # background w, h (150, 105) gemessen
        scale = 5
        w, h = 105, 148
        ######################################

        im = cv2.imread(PATH_TO_IMGS + "depth_map_hsv.bmp")
        # normalize image
        hh, ww = im.shape[:2]
        im_ = np.zeros((hh, ww))
        im = cv2.normalize(im, im_, 0, 255, cv2.NORM_MINMAX)


        measure = Measure(True)
        imgArray = []
        measure.black_frame = create_black_frame(ww, hh)
        while True:
            # GET BACKGROUND (working area)
            show_background = True
            if show_background:
                imm = im.copy()
                im_, contours, _ = measure.get_contours(
                    img=imm,
                    min_area=10000,
                    filter=4,
                    show_canny=True,
                    draw=True,
                    background=True,
                )
                # background
                background, _ = measure.get_background(contoures=contours)
                print(background)

                if background is None:
                    continue

                mask = np.zeros((hh, ww), np.uint8)
                # invert white mask
                # mask.fill(255)

                if len(contours) == 0:
                    continue

                # draw backgrownd contours
                con = contours[0]["cnt"]
                print("corners", contours[0]["corner"])
                mask = cv2.fillConvexPoly(mask, con, (255, 255, 255))
                cv2.waitKey(1)

                immm = stack_images(
                    scale=0.15, imgArray=[[measure.black_frame, im_, mask]]
                )
                hhh, www = immm.shape[:2]
                cv2.imshow(f"origin, mask, masked (w={www/2}, h={hhh})", immm)

            # TEST
            TEST = False
            if TEST:
                measure.get_contours_test(img=im)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()
