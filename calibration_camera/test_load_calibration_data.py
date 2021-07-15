"""Try to load the saved calibration parameters."""

import os
import numpy as np
import cv2 as cv


if __name__ == "__main__":
    load = np.load(os.path.join(os.path.dirname(__file__), "matrix_distortion.npz"))
    print(load)
    print(load["matrix"])
    print(load["distortion"])

    mtx = load["matrix"]
    dist = load["distortion"]
    num = int(input("Input camera number:"))
    cap = cv.VideoCapture(num)
    while cv.waitKey(1) != ord("q"):
        _, img = cap.read()
        h, w = img.shape[:2]
        new_camera_mtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        # undistort
        dst = cv.undistort(img, mtx, dist, None)
        # crop the image
        x, y, w, h = roi
        # dst = dst[y : y + h, x : x + w]
        cv.imshow("", dst)
