import cv2
import numpy as np


def getCorner(img):
    src = cv2.imread(img)
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    out = cv2.inRange(hsv, (0, 0, 183), (180, 50, 255))
    cv2.imshow('out', out)

    fh, fw = out.shape
    miny = fw
    # TODO:最好需要判断一下旁边的点的情况,万一会出现误差点
    miny = 1000
    # 寻找最下方的两个点
    for i in range(2, fh - 2):
        for j in range(2, fw - 2):
            if (out[i][j] == 255 and j < miny and out[i + 1][j] == 0 and out[i][j - 1] == 0 and out[i - 1][
                j + 1] == 255 and j < (1 / 2 * fw) and i > (fh / 2)):
                miny = j
                flag = 1
                minx = i
    maxy = 0
    for i in range(fh - 3):
        for j in reversed(range(2, fw - 3)):
            if i > (2 / fh) and j > (2 / fw):
                if (out[i][j] == 255 and j > maxy and out[i + 1][j] == 0 and out[i][j + 1] == 0 and out[i - 1][
                    j - 1] == 255 and j > (1 / 2 * fw) and i > (fh / 2)):
                    maxy = j
                    flag = 1
                    maxx = i
    # 寻找上方的两个点
    x1 = fh
    x2 = fh
    y1 = 0
    y2 = 0

    # index = []
    # point = []
    # for i in range(fh):
    #     for j in range(fw):
    #         point.append(1)
    #         index.append(1)
    #         for k in range(j, fw):
    #             if out[i][k] == 255:
    #                 point[-1] = point[-1] + 1
    #         break
    #
    # print(point)
    # for i in range(fh):
    #     if point[i] > 150 and point[i] - point[i - 1] > 50:
    #         print(i)

    # for i in range(2, fh):
    #     for j in range(2, fw):
    #         if (out[i][j] == 255):
    #             if (j < fw / 2 and i < (fh / 3 * 2)):
    #                 if (out[i - 1][j + 1] == 0 and out[i][j - 1] == 0 and out[i + 1][
    #                     j] == 255 and out[i][j + 1] == 255 and out[i + 2][j + 2] == 255 and out[i + 1][j + 1] == 255 and
    #                         out[i + 3][j - 3] == 0 and out[i + 2][j] == 255 and out[i + 2][j + 1] == 255 and out[i + 3][
    #                             j - 1] == 255 and out[i - 1][j - 2] == 0 and out[i - 1][j - 3] == 0 and out[i + 1][
    #                             j + 3] == 255 and j > miny):
    #                     if i <= x1:
    #                         x1 = i
    #                         y1 = j
    #                         break

    oflag = 0
    find = 0
    size = 6
    for i in range(size, fh - size):
        for j in range(size, fw - size):
            if (out[i][j] == 255):
                if (j < fw / 2 and i < (fh / 3 * 2) and out[i - 1][j + 1] == 0 and out[i][j - 1] == 0 and out[i - 1][
                    j - 2] == 0 and out[i - 1][j - 3] == 0 and out[i+1][j-3]==0 and out[i+1][j-2]==0 and (j - miny) < 180):
                    count = 0
                    for q in range(size):
                        for w in range(size):
                            if (out[i + 1 + q][j + w] == 255):
                                count = count + 1
                    myx = i
                    # print(i, j)
                    for k in reversed(range(fw)):
                        if out[i + 6][k] == 255:
                            myx = k
                            break
                    if count == size*size and (myx - j) > 300:
                        x1 = i
                        y1 = j
                        find = 1
                break

        if find == 1:
            break

    find = 0
    # last = 0
    # for i in range(2, fh-4):
    #     for j in reversed(range(2, fw-4)):
    #         if (out[i][j] == 255):
    #             if (j >= fw / 2 and i < (fh / 3 * 2)):
    #                 if (out[i - 1][j] == 0 and out[i - 1][j - 1] == 0 and out[i - 2][j - 2] == 0 and out[i + 1][
    #                     j + 2] == 0 and out[i][
    #                         j + 2] == 0 and out[i][j + 3] == 0 and out[i][
    #                         j + 1] == 0 and out[i + 1][j] == 255 and out[i + 2][j - 2] == 255 and out[i][
    #                         j - 1] == 255 and
    #                     out[i + 3][j] == 255 and out[i + 1][j - 1] == 255
    #                     and out[i - 2][j - 4] == 0 and j < maxy) and abs(i - x1) < 10:
    #                     last = last + 1
    #                     if i <= x2:
    #                         x2 = i
    #                         y2 = j

    oflag = 1
    find = 0
    for i in range(size, fh - size):
        for j in reversed(range(size, fw - size)):
            if (out[i][j] == 255):
                if (j >= fw / 2 and i < (fh / 3 * 2) and out[i - 1][j - 1] == 0 and out[i][j + 1] == 0 and
                        out[i - 1][j + 2] == 0 and out[i - 1][j + 3] == 0 and out[i+1][j+3]==0 and out[i+1][j+2]==0 and (maxy - j) < 180):
                    count = 0
                    for q in range(size):
                        for w in range(size):
                            if (out[i + 1 + q][j - w] == 255):
                                count = count + 1
                    myx = i
                    # print(i, j)
                    for k in range(fw):
                        if out[i + 6][k] == 255:
                            myx = k
                            #print(myx)
                            break
                    if count == size*size and (j - myx) > 300:
                        x2 = i
                        y2 = j
                        find = 1
                break
        if find == 1:
            break

    # print((miny, minx), (maxy, maxx), (y1, x1), (y2, x2))
    return (miny, minx), (maxy, maxx), (y1, x1), (y2, x2)
