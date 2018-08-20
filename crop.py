import cv2
import numpy as np
import math

def cross_point(x1, y1, x2, y2, x3, y3, x4, y4):  # 计算交点函数

    k1 = (y2 - y1) * 1.0 / (x2 - x1)  # 计算k1,由于点均为整数，需要进行浮点数转化
    b1 = y1 * 1.0 - x1 * k1 * 1.0  # 整型转浮点型是关键
    if (x4 - x3) == 0:  # L2直线斜率不存在操作
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)  # 斜率存在操作
        b2 = y3 * 1.0 - x3 * k2 * 1.0
    if k2 == None:
        x = x3
    else:
        x = (b2 - b1) * 1.0 / (k1 - k2)
    y = k1 * x * 1.0 + b1 * 1.0
    return [x, y]


image = cv2.imread('/home/yingkai/Data/mydata/item/images/train/item22.jpg')
who_edge = edges = cv2.Canny(image, 50, 150, apertureSize=3)
(im_height, im_width, chanel) = image.shape
hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
hsv = cv2.inRange(hsv, (0, 0, 128), (180, 30, 255))
minx = (int)(0.44579485058784485 * im_height - 10)
maxx = (int)(0.626618504524231 * im_height + 10)
miny = (int)(0.38149574398994446 * im_width - 10)
maxy = (int)(0.5359671115875244 * im_width + 10)

crop_image = image[minx:maxx, miny:maxy]
img = cv2.GaussianBlur(crop_image, (3, 3), 0)
edges = cv2.Canny(img, 50, 150, apertureSize=3)

(h, w) = edges.shape
print(h)
print(w)
count = 0
flag = 0
for i in reversed(range(h)):
    if flag == 0:
        for j in range(w):
            if (edges[i][j] == 255):
                lbx = i
                lby = j - 1
                flag = 1
                count = count + 1
                break
    else:
        break
flag = 0
for i in reversed(range(h)):
    if flag == 0:
        for j in reversed(range(w)):
            if (edges[i][j] == 255):
                rbx = i
                rby = j + 1
                flag = 1
                count = count + 1
                break
    else:
        break

print(count)
print(lbx)
print(lby)

print(minx + lbx)
print(miny + lby)

lf = 60
rf = 20
for i in reversed(range(lbx)):
    if (edges[i][lby - lf] == 255):
        print(i)
        temp = i
        break

for i in reversed(range(rbx)):
    if (edges[i][rby + rf] == 255):
        print(i)
        temp1 = i
        break

(myx, myy) = cross_point(miny + lby, minx + lbx,
                         miny + lby - lf, minx + temp,
                         miny + rby, minx + rbx,
                         miny + rby + rf, minx + temp1)
print(myx)
print(myy)
myx = round(myx)
myy = round(myy)
cv2.line(image, (myx, myy), (miny + lby - lf, minx + temp), (0, 255, 255))
cv2.line(image, (myx, myy), (miny + rby + rf, minx + temp1), (0, 255, 255))
# cv2.circle(image, ((myx), (myy)), 1, (0, 255, 255),1)


canvas = np.zeros((im_height, im_width, 1), dtype="uint8")
cv2.line(canvas, (myx, myy), (miny + lby - lf, minx + temp), 255)
cv2.line(canvas, (myx, myy), (miny + rby + rf, minx + temp1), 255)
# cv2.line(canvas, (miny + lby, minx + lbx), (miny + lby - 40, minx + temp), 255)
# cv2.line(canvas, (miny + rby, minx + rbx), (miny + rby + 20, minx + temp1), 255)
# cv2.circle(canvas, ((myx), (myy)), 1, 255)

cv2.imshow('edge', who_edge)
cv2.imshow('hsv', hsv)
cv2.imshow('src', image)
cv2.imshow('edges', edges)
cv2.imshow('crop', crop_image)
cv2.imwrite('out.jpg', image)

pts3 = np.float32([[286, 197], [623, 200], [145, 444], [694, 457]])
pts4 = np.float32([[0, 0], [im_height - 1, 0], [0, im_width - 161], [im_height - 1, im_width - 161]])
M_perspective = cv2.getPerspectiveTransform(pts3, pts4)
img_perspective = cv2.warpPerspective(image, M_perspective, (im_height, im_width - 160))
img_perspective = cv2.resize(img_perspective, (550, 550))
f_edges = cv2.Canny(img_perspective, 50, 150, apertureSize=3)
cv2.imshow('f_edge', f_edges)

# after_perspective = cv2.inRange(img_perspective, (54, 254, 154), (0, 255, 255))
# cv2.imshow('after_perspective', after_perspective)

perspective = cv2.warpPerspective(canvas, M_perspective, (im_height, im_width - 160))
perspective = cv2.resize(perspective, (550, 550))

fh = 550
fw = 550
for i in range(fh):
    for j in range(fw):
        if perspective[i][j] < 80:
            perspective[i][j] = 0

flag = 0
for i in reversed(range(fh)):
    if flag == 0:
        for j in range(fw):
            if (perspective[i][j] != 0):
                bx = i
                by = j
                flag = 1
                break
    else:
        break
miny = 500
for i in range(fh):
    for j in range(fw):
        if (perspective[i][j] != 0 and j < miny):
            miny = j
            flag = 1
            minx = i
maxy = 0
for i in range(fh):
    for j in reversed(range(fw)):
        if (perspective[i][j] != 0 and j > maxy):
            maxy = j
            flag = 1
            maxx = i

print("最底部点坐标为")
print(bx, by)
print("最左边坐标为", minx, miny)
print("最右边坐标为", maxx, maxy)
b1 = ((miny - by) * (miny - by) + (minx - bx) * (minx - bx)) ** 0.5
b2 = ((maxy - by) * (maxy - by) + (maxx - bx) * (maxx - bx)) ** 0.5
print("线段长为",b1,b2)
wenjuhc = 300
wenjuhk = 100
x3 = bx - (wenjuhc * (bx - minx) / b1)
y3 = by - (wenjuhc * (by - miny) / b1)
x4 = bx + (wenjuhk * (maxx - bx) / b2)
y4 = by - (wenjuhk * (by - miny) / b2)

print("x3坐标为",x3, y3)
print("x4坐标为",x4, y4)
print("中心坐标为",(x3+x4)/2,(y3+y4)/2)
theta = math.asin((by-miny)/b1)*180/3.1415
print("角度为",theta)
cv2.imshow('rs', perspective)
cv2.imshow('output', canvas)
cv2.imshow('perspective', img_perspective)
cv2.waitKey()
