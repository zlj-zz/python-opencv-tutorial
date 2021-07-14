import numpy as np
import cv2 as cv


class HSVColor:

    def __init__(self):
        pass

    red_low = np.array([0, 43, 46])
    green_low = np.array([35, 43, 46])
    blue_low = np.array([100, 43, 46])

    red_high = np.array([10, 255, 255])
    green_high = np.array([77, 255, 255])
    blue_high = np.array([124, 255, 255])


class METER:

    def __init__(self, image):
        """

        :param image:
        """
        if isinstance(image, str):
            self.processed = self.image = cv.imread(image)
        elif isinstance(image, np.ndarray):
            self.image = image
            self.processed = image

        self.canny = None
        self.contours = None
        self.cnt = None
        self.wr = []
        self.pi = []

    def colorChange(self, mode):
        """

        :param mode: "gray" or "hsv"
        :return:
        """
        if mode == "gray":
            self.processed = cv.cvtColor(self.processed, cv.COLOR_BGR2GRAY)
        elif mode == "hsv":
            self.processed = cv.cvtColor(self.processed, cv.COLOR_BGR2HSV)
        else:
            print("colorchanged is wrong!\n", exit("dammit"))
        return self.processed

    def color_trace(self, color):
        mask = None
        if color == "red":
            mask = cv.inRange(self.processed, HSVColor.red_low,
                              HSVColor.red_high)
        elif color == "blue":
            mask = cv.inRange(self.processed, HSVColor.blue_low,
                              HSVColor.blue_high)
        elif color == "green":
            mask = cv.inRange(self.processed, HSVColor.green_low,
                              HSVColor.green_high)
        else:
            exit("color_trace is wrong")
        erosion = cv.erode(mask, np.ones((1, 1), np.uint8), iterations=2)
        dilation = cv.dilate(erosion, np.ones((1, 1), np.uint8), iterations=2)
        target = cv.bitwise_and(self.image, self.image, mask=dilation)

        # 将滤波后的图像变成二值图像放在binary中
        ret, binary = cv.threshold(dilation, 127, 255, cv.THRESH_BINARY)
        # 在binary中发现轮廓，轮廓按照面积从小到大排列
        contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL,
                                              cv.CHAIN_APPROX_SIMPLE)
        m = 0
        for i in contours:
            x, y, w, h = cv.boundingRect(i)   # 将轮廓分解为识别对象的左上角坐标和宽、高
            # 在图像上画上矩形（图片、左上角坐标、右下角坐标、颜色、线条宽度）
            if min(self.image.shape[0], self.image.shape[1]) / 10 < min(h, w) \
                    < min(self.image.shape[0], self.image.shape[1]) / 1:
                cv.rectangle(self.image, (x, y), (x + w, y + h), (
                    0,
                    255,
                ), 3)
                # 给识别对象写上标号
                font = cv.FONT_HERSHEY_SIMPLEX
                cv.putText(self.image, color, (x - 10, y + 10), font, 1,
                           (0, 0, 255), 2)   # 加减10是调整字符位置
        return self.image


def color_trace(image, color):
    Trace = METER(image)
    Trace.colorChange("hsv")
    return Trace.color_trace(color)


def trace_color_in_video():
    cap = cv.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        out = color_trace(frame, "red")
        cv.imshow('frame', out)
        if cv.waitKey(1) == 27:
            break
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':

    trace_color_in_video()
