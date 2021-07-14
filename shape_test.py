import cv2
import numpy as np

def detect_circle(image):
    output = image
    # 将其转换为灰度图片
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 应用hough变换进行圆检测
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)

    # 确保至少发现一个圆
    if circles is not None:
            # 进行取整操作
            circles = np.round(circles[0, :]).astype("int")

            # 循环遍历所有的坐标和半径
            for (x, y, r) in circles:
                    # 绘制结果
                    cv2.circle(output, (x, y), r, (0, 255, 0), 4)
                    cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

            # 显示结果
            # cv2.imshow("output", np.hstack([image, output]))
            cv2.imshow("output", output)
            # cv2.waitKey(0)

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        detect_circle(frame)
        # cv2.imshow('img', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    pass
