
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

# assumption: the paper is upright and is a rectangular figure
# returns the paper in a black and white format

# read image in black and white
directory = r"C:\Users\OWNER\Downloads\python\OpenCV\images\perspective transformation samples"


def DocumentScanner(address):
    img = cv.imread(address)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # find the largest contour (the box) using manually implemented canny algorithm
    # get kernel size
    xy = []
    for scale in gray.shape:
        xy.append(scale // 100) if scale // 100 % 2 else xy.append(scale // 100 + 1)
    kernel = tuple(xy)

    # thresh, blur, and find contours
    thresh = cv.threshold(gray, 160, 255, cv.THRESH_BINARY)[1]
    # thresh = cv.threshold(gray, 0, 255, cv.THRESH_OTSU)[1]
    blur = cv.GaussianBlur(thresh, kernel, 0)
    cnt = cv.findContours(blur, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)[0]
    if not len(cnt):
        raise ValueError("No contours found")
    # c = max(cnt, key=cv.contourArea)
    # either simply find the max or find largest contour with four points using approximation
    cnts = sorted(cnt, key=cv.contourArea, reverse=True)[:30]
    c = None
    for contour in cnts:
        peri = cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, peri * 0.03, True)
        if len(approx) == 4:
            c = approx
            break

    # find the four extreme points of the largest contour c
    topLeft = c[c[:, :, 0].argmin()][0]
    botRight = c[c[:, :, 0].argmax()][0]
    topRight = c[c[:, :, 1].argmin()][0]
    botLeft = c[c[:, :, 1].argmax()][0]

    extremePoints = np.array(
        [topLeft, botRight, topRight, botLeft], dtype=np.float32)

    # calculate the new width using pythagorean theorem
    x1 = np.diff([topLeft[0], topRight[0]])
    y1 = np.diff([topLeft[1], topRight[1]])
    x2 = np.diff([botLeft[0], botRight[0]])
    y2 = np.diff([botLeft[1], botRight[1]])

    w = int(max(np.sqrt(x1 ** 2 + y1 ** 2)[0], np.sqrt(x2 ** 2 + y2 ** 2)[0]))

    x1 = np.diff([topRight[0], botRight[0]])
    y1 = np.diff([topRight[1], botRight[1]])
    x2 = np.diff([topLeft[0], botLeft[0]])
    y2 = np.diff([topLeft[1], botLeft[1]])

    h = int(max(np.sqrt(x1 ** 2 + y1 ** 2)[0], np.sqrt(x2 ** 2 + y2 ** 2)[0]))

    # destination size
    dst = np.array([[0, 0],
                    [w-5, h-5],
                    [w-5, 0],
                    [0, h-5]], dtype="float32")

    M = cv.getPerspectiveTransform(extremePoints, dst)
    warped = cv.warpPerspective(thresh, M, (w, h))

    plt.figure(figsize=[15, 15])
    plt.subplot(121)
    plt.imshow(img)
    plt.title("original")
    plt.subplot(122)
    plt.imshow(warped, cmap='gray')
    plt.title("warped")
    plt.waitforbuttonpress()
    plt.close("all")


for photo in os.listdir(directory):
    address = directory + "\\" + photo
    DocumentScanner(address)
