#!/usr/bin/env python3

"""Generate the grip image for calibration."""

import os
import cv2


path = os.path.join(os.path.dirname(__file__), "3a4.bmp")
print(path)

frame = cv2.imread(path)
# print(frame)
row, col, nc = frame.shape

width_of_roi = 90

for j in range(row):
    data = frame[j]
    for i in range(col):
        f = int(i / width_of_roi) % 2 ^ int(j / width_of_roi) % 2
        if f:
            frame[j][i][0] = 255
            frame[j][i][1] = 255
            frame[j][i][2] = 255
cv2.imshow("", frame)
cv2.waitKey(0) & 0xFF == ord("q")
# cv2.imwrite('./calibration.jpg',frame)
