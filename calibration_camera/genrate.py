import os
import cv2

dir_ = os.path.dirname(__file__)


frame = cv2.imread(os.path.join(dir_, "3a4.bmp"))
# print(frame)
row, col, nc = frame.shape

nWidthOfROI = 90

for j in range(row):
    data = frame[j]
    print(data, len(data))
    for i in range(col):
        f = int(i / nWidthOfROI) % 2 ^ int(j / nWidthOfROI) % 2
        # print(f)
        if f:
            frame[j][i][0] = 255
            frame[j][i][1] = 255
            frame[j][i][2] = 255
cv2.imshow("", frame)
cv2.waitKey(0)
# cv2.imwrite('./calibration.jpg',frame)
