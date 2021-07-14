import cv2
import numpy as np


def stack_images(imgArray, scale, lables=[]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        hor_con = np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth = int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        print(eachImgHeight)
        for d in range(0, rows):
            for c in range(0, cols):
                cv2.rectangle(
                    ver,
                    (c * eachImgWidth, eachImgHeight * d),
                    (
                        c * eachImgWidth + len(lables[d]) * 13 + 27,
                        30 + eachImgHeight * d,
                    ),
                    (255, 255, 255),
                    cv2.FILLED,
                )
                cv2.putText(
                    ver,
                    lables[d],
                    (eachImgWidth * c + 10, eachImgHeight * d + 20),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.7,
                    (255, 0, 255),
                    2,
                )
    return ver


def nothing(x):
    pass


def initializeTrackbars():
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)
    cv2.createTrackbar("Threshold1", "Trackbars", 200, 255, nothing)
    cv2.createTrackbar("Threshold2", "Trackbars", 200, 255, nothing)


def valTrackbars():
    Threshold1 = cv2.getTrackbarPos("Threshold1", "Trackbars")
    Threshold2 = cv2.getTrackbarPos("Threshold2", "Trackbars")
    return Threshold1, Threshold2


def biggest_contour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 5000:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area


def draw_rectangle(img, biggest, thickness):
    cv2.line(
        img,
        (biggest[0][0][0], biggest[0][0][1]),
        (biggest[1][0][0], biggest[1][0][1]),
        (0, 255, 0),
        thickness,
    )
    cv2.line(
        img,
        (biggest[0][0][0], biggest[0][0][1]),
        (biggest[2][0][0], biggest[2][0][1]),
        (0, 255, 0),
        thickness,
    )
    cv2.line(
        img,
        (biggest[3][0][0], biggest[3][0][1]),
        (biggest[2][0][0], biggest[2][0][1]),
        (0, 255, 0),
        thickness,
    )
    cv2.line(
        img,
        (biggest[3][0][0], biggest[3][0][1]),
        (biggest[1][0][0], biggest[1][0][1]),
        (0, 255, 0),
        thickness,
    )

    return img


def reorder(my_points):
    my_points = my_points.reshape((4, 2))
    my_points_new = np.zeros((4, 1, 2), dtype=np.int32)

    add = my_points.sum(1)
    my_points_new[0] = my_points[np.argmin(add)]
    my_points_new[3] = my_points[np.argmax(add)]
    diff = np.diff(my_points, axis=1)
    my_points_new[1] = my_points[np.argmin(diff)]
    my_points_new[2] = my_points[np.argmax(diff)]

    return my_points_new


cap = cv2.VideoCapture(0)
cap.set(10, 160)  # light
height_img = 640
width_img = 480

initializeTrackbars()
count = 0

while True:
    blank_img = np.zeros((height_img, width_img, 3), np.uint8)
    success, img = cap.read()
    img = cv2.resize(img, (width_img, height_img))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
    threshold = valTrackbars()  # get track bar values for thresholds
    img_threshold = cv2.Canny(img_blur, threshold[0], threshold[1])
    kernel = np.ones((5, 5))
    img_dial = cv2.dilate(img_threshold, kernel, iterations=2)  # apply dilation
    img_threshold = cv2.erode(img_dial, kernel, iterations=1)  # apply erosion

    img_contours = img.copy()
    img_big_contour = img.copy()
    contors, hierarchy = cv2.findContours(
        img_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(img_contours, contors, -1, (0, 255, 0), 10)

    # Find the biggest contours
    biggest, max_area = biggest_contour(contors)
    if biggest.size != 0:
        biggest = reorder(biggest)
        cv2.drawContours(img_big_contour, biggest, -1, (0, 255, 0), 20)
        img_big_contour = draw_rectangle(img_big_contour, biggest, 2)
        pts1 = np.float32(biggest)
        pts2 = np.float32(
            [[0, 0], [width_img, 0], [0, height_img], [width_img, height_img]]
        )
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        img_warp_colored = cv2.warpPerspective(img, matrix, (width_img, height_img))

        # Remove 20 pixels form each side
        img_warp_colored = img_warp_colored[
            20 : img_warp_colored.shape[0] - 20, 20 : img_warp_colored.shape[1] - 20
        ]
        img_warp_colored = cv2.resize(img_warp_colored, (width_img, height_img))

        # apply adaptive threshold
        img_warp_gray = cv2.cvtColor(img_warp_colored, cv2.COLOR_BGR2GRAY)
        img_adaptive_threshold = cv2.adaptiveThreshold(img_warp_gray, 255, 1, 1, 7, 2)
        img_adaptive_threshold = cv2.bitwise_not(img_adaptive_threshold)
        img_adaptive_threshold = cv2.medianBlur(img_adaptive_threshold, 3)

        image_array = (
            [img, img_gray, img_threshold, img_contours],
            [img_big_contour, img_warp_colored, img_warp_gray, img_adaptive_threshold],
        )
    else:
        image_array = (
            [img, img_gray, img_threshold, img_contours],
            [blank_img, blank_img, blank_img, blank_img],
        )

    lables = [["a", "b", "c"], ["d", "e", "f", "g"]]
    stackedImage = stack_images(image_array, 0.75)
    cv2.imshow("Res", stackedImage)

    # SAVE IMAGE WHEN 's' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("s"):
        cv2.imwrite("Scanned/myImage" + str(count) + ".jpg", img_warp_colored)
        cv2.rectangle(
            stackedImage,
            (
                (int(stackedImage.shape[1] / 2) - 230),
                int(stackedImage.shape[0] / 2) + 50,
            ),
            (1100, 350),
            (0, 255, 0),
            cv2.FILLED,
        )
        cv2.putText(
            stackedImage,
            "Scan Saved",
            (int(stackedImage.shape[1] / 2) - 200, int(stackedImage.shape[0] / 2)),
            cv2.FONT_HERSHEY_DUPLEX,
            3,
            (0, 0, 255),
            5,
            cv2.LINE_AA,
        )
        cv2.imshow("Result", stackedImage)
        cv2.waitKey(300)
        count += 1
