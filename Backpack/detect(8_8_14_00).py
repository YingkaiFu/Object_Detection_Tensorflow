import numpy as np
import os
import pyrealsense2 as rs
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import math
# import corner

from PIL import Image
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

sys.path.append("..")
from object_detection.utils import ops as utils_ops


from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util


def getCanny(id):
    if id == '1':
        return 44, 18
    elif id == '2':
        return 28, 3
    elif id == '3':
        return 127, 0
    elif id == '4':
        return 33, 23
    elif id == '5':
        return 30, 45
    elif id == '6':
        return 30, 6
    elif id == '7':
        return 50, 0
    elif id == '8':
        return 58, 0
    elif id == '9':
        return 36, 82
    elif id == '10':
        return 34, 67
    elif id == '11':
        return 71, 14
    elif id == '12':
        return 93, 11
    elif id == '13':
        return 48, 19
    elif id == '14':
        return 29, 25
    elif id == '15':
        return 38, 0
    elif id == '16':
        return 59, 0
    elif id == '17':
        return 59, 0
    elif id == '18':
        return 71, 0
    elif id == '19':
        return 47, 0
    elif id == '20':
        return 24, 0


def getid(name):
    if name == 'xyy':
        return '1'
    elif name == 'pm':
        return '2'
    elif name == 'xb':
        return '3'
    elif name == 'yz':
        return '4'
    elif name == 'wjh':
        return '5'
    elif name == 'w':
        return '6'
    elif name == 'qq':
        return '7'
    elif name == 'yg':
        return '8'
    elif name == 'mj':
        return '9'
    elif name == 'kxt':
        return '10'
    elif name == 'jd':
        return '11'
    elif name == 's':
        return '12'
    elif name == 'gz':
        return '13'
    elif name == 'nn':
        return '14'
    elif name == 'z':
        return '15'
    elif name == 'kc':
        return '16'
    elif name == 'bg':
        return '17'
    elif name == 'ys':
        return '18'
    elif name == 'sp':
        return '19'
    elif name == 'st':
        return '20'


def ifsquare(name):
    if name == 'xyy' or name == 'yz' or name == 'gz':
        return 0
    elif name == 'pm' or name == 'xb' or name == 'w' or name == 'qq' or name == 'kxt' or name == 'jd' or name == 'st':
        return 1
    else:
        return 2


def getitem(id):
    if id == '1':
        return 210, 10
    elif id == '2':
        return 72
    elif id == '3':
        return 60
    elif id == '4':
        return 83, 56
    elif id == '5':
        return 204, 80
    elif id == '6':
        return 110
    elif id == '7':
        return 151
    elif id == '8':
        return 235, 54
    elif id == '9':
        return 192, 165
    elif id == '10':
        return 47
    elif id == '11':
        return 95
    elif id == '12':
        return 215, 155
    elif id == '13':
        return 130, 57
    elif id == '14':
        return 65, 43
    elif id == '15':
        return 205, 109
    elif id == '16':
        return 125, 67
    elif id == '17':
        return 150, 95
    elif id == '18':
        return 230, 70
    elif id == '19':
        return 250, 185
    elif id == '20':
        return 60


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


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                print(detection_boxes)
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


def main(itemId=[], positionx=[], positiony=[], Theta=[]):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    # config.enable_stream(rs.stream.depth, 960, 540, rs.format.z16, 30)
    profile = pipeline.start(config)
    #
    t = 1
    # depth_sensor = profile.get_device().first_depth_sensor()
    # depth_scale = depth_sensor.get_depth_scale()
    #
    # clipping_distance_in_meters = 1.2  # 1 meter
    # clipping_distance = clipping_distance_in_meters / depth_scale
    #
    # align_to = rs.stream.color
    # align = rs.align(align_to)
    #
    # # 相机刚开始工作的时候色温不对,因此需要进行预热
    while True:

        frames = pipeline.wait_for_frames()
        # aligned_frames = align.process(frames)
        # aligned_depth_frame = aligned_frames.get_depth_frame()
        # aligned_depth_frame is a 640x480 depth image

        color_frame = frames.get_color_frame()
        #
        #     # Convert images to numpy arrays
        #     depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        #     # Remove background - Set pixels further than clipping_distance to grey
        #     grey_color = 153
        #     depth_image_3d = np.dstack(
        #         (depth_image, depth_image, depth_image))  # depth image is 1 channel, color is 3 channels
        #     bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
        #
        #     # Render images
        #     depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        #
        if t == 100:
            cv2.imwrite('img.jpg', color_image)
            # images_stack = np.hstack((bg_removed, depth_colormap))
            break
        t = t + 1

    pipeline.stop()
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    # PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
    PATH_TO_CKPT = '/home/yingkai/frozen_inference_graph.pb'
    # List of the strings that is used to add correct label for each box.
    # PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
    PATH_TO_LABELS = '/home/yingkai/label.pbtxt'
    NUM_CLASSES = 20

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    name = 'img.jpg'
    image_path = name
    image = Image.open(image_path)
    (im_width, im_height) = image.size
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)

    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=1)
    img1 = cv2.cvtColor(np.asarray(image_np), cv2.COLOR_RGB2BGR)
    cv2.imshow("result", img1)
    cv2.waitKey()

    for i in range(output_dict['detection_boxes'].shape[0]):
        if output_dict['detection_scores'] is None or output_dict['detection_scores'][i] > 0.8:
            item = category_index[output_dict['detection_classes'][i]]['name']
            box = tuple(output_dict['detection_boxes'][i].tolist())
            # print(output_dict['detection_boxes'][i][1])
            print(item, output_dict['detection_scores'][i])
            # print(box)

            image = cv2.imread(name)
            who_edge = edges = cv2.Canny(image, 50, 150, apertureSize=3)
            (im_height, im_width, chanel) = image.shape
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

            # TODO: 手动阈值,比赛的时候需要根据实际情况进行调整
            hsv = cv2.inRange(hsv, (0, 0, 128), (180, 30, 255))

            minx = (int)(output_dict['detection_boxes'][i][0] * im_height)
            maxx = (int)(output_dict['detection_boxes'][i][2] * im_height * 1.025)
            miny = (int)(output_dict['detection_boxes'][i][1] * im_width)
            maxy = (int)(output_dict['detection_boxes'][i][3] * im_width)

            # 对图片进行裁剪,得到需要的物体区域框,然后进行滤波和边缘检测
            crop_image = image[minx:maxx, miny:maxy]
            crop_image = cv2.GaussianBlur(crop_image, (3, 3), 0)
            lowCanny, highCanny = getCanny(getid(item))
            edges = cv2.Canny(crop_image, 0 , 150, apertureSize=3)

            # 找到最底部最左边的点和最右边的点,这里考虑是因为可能底部不会是尖的而是一条直线
            (h, w) = edges.shape
            # print(h)
            # print(w)
            flag = 0
            for i in reversed(range(h - 1)):
                if flag == 0:
                    for j in range(w):
                        if (edges[i][j] == 255 and edges[i + 1][j] == 0):
                            lbx = i
                            lby = j
                            flag = 1
                            break
                else:
                    break
            flag = 0
            for i in reversed(range(h - 1)):
                if flag == 0:
                    for j in reversed(range(w)):
                        if (edges[i][j] == 255 and edges[i + 1][j] == 0):
                            rbx = i
                            rby = j
                            flag = 1
                            break
                else:
                    break
            print('两个底部点的y坐标差为',rby-lby)
            print(lbx, lby)
            print(rbx, rby)

            # TODO: 需要找到桌子的四个定点,然后输入到下面的程序进行透视变换
            # (x1,y1),(x2,y2),(x3,y3),(x4,y4) = corner.getCorner(hsv)
            pts3 = np.float32([[260, 156], [612, 162], [123, 473], [719, 475]])
            # pts3 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            pts4 = np.float32([[0, 0], [550, 0], [0, 550], [550, 550]])
            M_perspective = cv2.getPerspectiveTransform(pts3, pts4)

            # 桌子的尺寸
            fh = 550
            fw = 550

            if ifsquare(item) == 2:

                # 相对于原图的坐标
                print('最底部点相对于原图的坐标为', minx + lbx, miny + lby)
                lf = round(0.4 * (lby))
                rf = round(0.4 * (w - rby))
                if rby-lby<8:
                    print('最底部点位于裁剪图的',(lby)/(maxy-miny))
                    lf = round(0.7*lby)
                    rf = round((maxy-miny-rby)*0.3)

                cv2.imshow('edges', edges)
                cv2.imshow('crop', crop_image)
                cv2.imwrite('out.jpg', image)
                cv2.waitKey()
                # 从最下的点开始往左搜索固定长度,如果上方有白点,则进行连线,此处即为底部轮廓的一条边
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

                # 求两条直线的交点
                (myx, myy) = cross_point(miny + lby, minx + lbx,
                                         miny + lby - lf, minx + temp,
                                         miny + rby, minx + rbx,
                                         miny + rby + rf, minx + temp1)
                myx = round(myx)
                myy = round(myy)
                # print(myx)
                # print(myy)

                # 在原图上画线段,
                cv2.line(image, (myx, myy), (miny + lby - lf, minx + temp), (0, 255, 255))
                cv2.line(image, (myx, myy), (miny + rby + rf, minx + temp1), (0, 255, 255))
                # cv2.circle(image, ((myx), (myy)), 1, (0, 255, 255),1)

                # 新建一个图层来进行线段的绘画,使幕板图为灰度图
                canvas = np.zeros((im_height, im_width, 1), dtype="uint8")
                cv2.line(canvas, (myx, myy), (miny + lby - lf, minx + temp), 255)
                cv2.line(canvas, (myx, myy), (miny + rby + rf, minx + temp1), 255)
                # cv2.line(canvas, (miny + lby, minx + lbx), (miny + lby - 40, minx + temp), 255)
                # cv2.line(canvas, (miny + rby, minx + rbx), (miny + rby + 20, minx + temp1), 255)
                # cv2.circle(canvas, ((myx), (myy)), 1, 255)

                # 输出一波图像
                cv2.imshow('edge', who_edge)
                cv2.imshow('hsv', hsv)
                cv2.imshow('src', image)
                cv2.waitKey()
                # 220 294

                # 原图的透视变换
                img_perspective = cv2.warpPerspective(image, M_perspective, (550, 550))
                f_edges = cv2.Canny(img_perspective, 50, 150, apertureSize=3)
                cv2.imshow('f_edge', f_edges)

                # after_perspective = cv2.inRange(img_perspective, (54, 254, 154), (0, 255, 255))
                # cv2.imshow('after_perspective', after_perspective)

                # 幕板的透视变换,变换后缩放到桌子的长度,则图上1个像素点表示1mm
                perspective = cv2.warpPerspective(canvas, M_perspective, (550, 550))

                # 由于变换后的直线出现了形变,因此筛选处一些点
                for i in range(fh):
                    for j in range(fw):
                        if perspective[i][j] < 80:
                            perspective[i][j] = 0

                # 寻找最底部的点.
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

                # 寻找左边的线段点
                miny = 500
                for i in range(fh):
                    for j in range(fw):
                        if (perspective[i][j] != 0 and j < miny):
                            miny = j
                            flag = 1
                            minx = i

                # 寻找右边的线段点
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

                # 角度和长度的计算
                b1 = ((miny - by) * (miny - by) + (minx - bx) * (minx - bx)) ** 0.5
                b2 = ((maxy - by) * (maxy - by) + (maxx - bx) * (maxx - bx)) ** 0.5
                print("线段长为", b1, b2)
                # itemchang,itemkuan = getitem(getid(category_index[output_dict['detection_classes'][i]]['name']))
                if b1 > b2:
                    print('长边在左')
                    itemleft, itemright = (getitem(getid(item)))
                    newmaxy = by + itemright * (bx - minx) / itemleft
                    newmaxx = bx - itemright * (by - miny) / itemleft
                    cv2.line(perspective, (by, bx), (round(newmaxy),round(newmaxx)), 255)
                    print(newmaxx, newmaxy)
                else:
                    print('长边在右')
                    itemright, itemleft = (getitem(getid(item)))
                    newminy = by - itemleft * (bx - maxx) / itemright
                    newminx = bx - itemleft * (maxy - by) / itemright
                    cv2.line(perspective, (by, bx), (round(newminy), round(newminx)), 255)
                    print(newminx, newminy)

                print(itemleft, itemright)
                x3 = bx - (itemleft * (bx - minx) / b1)
                y3 = by - (itemleft * (by - miny) / b1)
                x4 = bx + (itemright * (maxx - bx) / b2)
                y4 = by - (itemright * (by - maxy) / b2)

                print("x3坐标为", x3, y3)
                print("x4坐标为", x4, y4)
                print('物体为', item, '编号为', getid(item))
                print("中心坐标为", (x3 + x4) / 2, (y3 + y4) / 2)
                theta = math.asin((by - miny) / b1) * 180 / math.pi
                print("角度为", 90 - theta)
                itemId.append(getid(item) + '  ')
                positionx.append(str(round((y3 + y4) / 2)) + '  ')
                positiony.append(str(round((x3 + x4) / 2)) + '  ')
                Theta.append(str(round(90 - theta)) + '  ')
                cv2.imshow('rs', perspective)
                cv2.imshow('output', canvas)
                cv2.imshow('edges', edges)
                cv2.imshow('perspective', img_perspective)
                cv2.waitKey()

            elif ifsquare(item) == 1:
                # 直接取中点来计算
                nx = round((lbx + rbx) / 2) + minx
                ny = round((lby + rby) / 2) + miny
                canvas = np.zeros((im_height, im_width, 1), dtype="uint8")
                cv2.circle(canvas, (ny, nx), 1, 255)
                cv2.circle(image, (ny, nx), 1, (255, 255, 0))
                cv2.circle(image, (lby, lbx), 1, (255, 255, 0))
                cv2.circle(image, (rby, rbx), 1, (255, 255, 0))
                img_perspective = cv2.warpPerspective(image, M_perspective, (550, 550))

                # img_perspective = cv2.GaussianBlur(img_perspective,(3,3),0)
                f_edges = cv2.Canny(img_perspective, 0, 150, apertureSize=3)
                cv2.imshow('f_edge', img_perspective)
                cv2.imshow('edges', edges)
                cv2.imshow('crop', crop_image)
                perspective = cv2.warpPerspective(canvas, M_perspective, (550, 550))
                cv2.imshow('rs', perspective)

                # 筛选掉一些点
                for i in range(fh):
                    for j in range(fw):
                        if perspective[i][j] < 80:
                            perspective[i][j] = 0

                # 寻找最底部的点.
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
                print(bx, by)
                bx = bx - getitem(getid(item)) / 2
                print(item, '的中心位置为', bx, by)
                itemId.append(getid(item) + '  ')
                positionx.append(str(round(bx)) + '  ')
                positiony.append(str(round(by)) + '  ')
                Theta.append('None' + '  ')
            elif ifsquare(item) == 0:
                by = maxy
                chang, kuan = getitem(getid(item))
                bx = maxx / 1.025
                canvas = np.zeros((im_height, im_width, 1), dtype="uint8")
                cv2.circle(canvas, (by, round(bx)), 1, 255)
                perspective = cv2.warpPerspective(canvas, M_perspective, (550, 550))
                # 筛选掉一些点
                for i in range(fh):
                    for j in range(fw):
                        if perspective[i][j] < 80:
                            perspective[i][j] = 0

                # 寻找最底部的点.
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
                print(item, '的中心位置为', bx, by)
    # plt.figure(figsize=IMAGE_SIZE)
    # plt.imshow(image_np)
    img1 = cv2.cvtColor(np.asarray(image_np), cv2.COLOR_RGB2BGR)
    cv2.imshow("result", img1)
    cv2.waitKey()
