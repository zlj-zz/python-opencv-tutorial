#!/usr/bin/env python

"""
Calibrate fish eye camera.
env:
    python2
    python3
"""

import os
import glob
import numpy as np
import cv2 as cv
from pprint import pformat


def left_indent(text, indent=4):
    indent_str = " " * indent
    return "".join((indent_str + l for l in text.splitlines(True)))


def calibration_camera(row, col, path=None, cap_num=None, saving=False):
    """Calibrate camera

    Args:
        row (int): The number of rows in the grid.
        col (int): The number of columns in the grid.
        path (string): Directory path for storing calibration pictures(.jpg).
        saving (bool): If to save the calibrated parameters(.npz).
    """

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    obj_p = np.zeros((row * col, 3), np.float32)
    obj_p[:, :2] = np.mgrid[0:row, 0:col].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.

    def _find_grid(img):
        _find_grid.gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(_find_grid.gray, (row, col), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            _find_grid.obj_points.append(obj_p)
            corners2 = cv.cornerSubPix(
                _find_grid.gray, corners, (11, 11), (-1, -1), criteria
            )
            _find_grid.img_points.append(corners)
            # Draw and display the corners
            cv.drawChessboardCorners(img, (row, col), corners2, ret)

    _find_grid.obj_points = []  # 3d point in real world space
    _find_grid.img_points = []  # 2d points in image plane.
    _find_grid.gray = None

    if path and cap_num:
        raise Exception("The parameter `path` and `cap_num` only need one.")
    if path:
        images = glob.glob(os.path.join(path, "cal_*.jpg"))
        print("Found calibration images:")
        print(left_indent(pformat(images)))
        print("-" * 70)
        for f_name in images:
            img = cv.imread(f_name)
            _find_grid(img)
            cv.imshow("img", img)
            cv.waitKey(500)
    if cap_num:
        cap = cv.VideoCapture(cap_num)
        while True:
            _, img = cap.read()
            _find_grid(img)
            cv.imshow("img", img)
            cv.waitKey(500)
            print(len(_find_grid.obj_points))
            if len(_find_grid.obj_points) > 14:
                break

    cv.destroyAllWindows()

    print("Found grid number: {}.".format(len(_find_grid.obj_points)))
    print("-" * 70)

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        _find_grid.obj_points,
        _find_grid.img_points,
        _find_grid.gray.shape[::-1],
        None,
        None,
    )
    print("Calibration result:")
    print("    ret: {}".format(ret))
    print(left_indent("matrix:\n{}".format(pformat(mtx))))
    print(left_indent("distortion: {}".format(pformat(dist))))
    print("-" * 70)
    if saving:
        print("Saving calibration data, path: <current_path>/matrix_distortion.npz")
        print("-" * 70)
        np.savez("matrix_distortion", matrix=mtx, distortion=dist)

    mean_error = 0
    for i in range(len(_find_grid.obj_points)):
        img_points_2, _ = cv.projectPoints(
            _find_grid.obj_points[i], rvecs[i], tvecs[i], mtx, dist
        )
        error = cv.norm(_find_grid.img_points[i], img_points_2, cv.NORM_L2) / len(
            img_points_2
        )
        mean_error += error
    print("Testing result:")
    print("    total error: {}".format(mean_error / len(_find_grid.obj_points)))
    print("-" * 70)

    return mtx, dist


if __name__ == "__main__":
    path = os.path.dirname(__file__)
    mtx, dist = calibration_camera(8, 6, path, saving=True)
    # mtx, dist = calibration_camera(8, 6, cap_num=1, saving=True)

    if_test = input("If testing the result (default: no), [yes/no]:")
    if if_test not in ["y", "Y", "yes", "Yes"]:
        exit(0)

    cap_num = int(input("Input camera number:"))
    cap = cv.VideoCapture(1)
    while cv.waitKey(1) != ord("q"):
        _, img = cap.read()
        h, w = img.shape[:2]
        new_camera_mtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        # undistort
        dst = cv.undistort(img, mtx, dist)
        # crop the image
        x, y, w, h = roi
        # dst = dst[y : y + h, x : x + w]
        cv.imshow("", dst)
