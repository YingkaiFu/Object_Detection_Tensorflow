import numpy as np
import os
import sys
import tensorflow as tf
import math
from Tools import corner

from PIL import Image
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

sys.path.append("..")
from object_detection.utils import ops as utils_ops

from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util


def judge_box(box, i, count):
    flag = []
    for j in range(count):
        if j == i:
            continue
        if (box[i][2] > box[j][0] and box[i][3] > box[j][1]
                and box[j][2] > box[i][0] and box[j][3] > box[i][1]):
            flag.append(j)
            print('the object is overlapped with ' + str(j) + ' item')
    return flag


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
        return 79
    elif id == '3':
        return 60
    elif id == '4':
        return 83, 56
    elif id == '5':
        return 204, 80
    elif id == '6':
        return 110
    elif id == '7':
        return 234
    elif id == '8':
        return 235, 54
    elif id == '9':
        return 192, 165
    elif id == '10':
        return 49
    elif id == '11':
        return 95
    elif id == '12':
        return 215, 155
    elif id == '13':
        return 130, 57
    elif id == '14':
        return 95, 63
    elif id == '15':
        return 205, 109
    elif id == '16':
        return 125, 67
    elif id == '17':
        return 120, 95
    elif id == '18':
        return 230, 70
    elif id == '19':
        return 235, 180
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
    # pipeline = rs.pipeline()
    # config = rs.config()
    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    # config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    #
    # profile = pipeline.start(config)
    # dev = profile.get_device()
    # sensor = dev.query_sensors()
    # depth_sensor = profile.get_device().first_depth_sensor()
    # depth_scale = depth_sensor.get_depth_scale()
    #
    # clipping_distance_in_meters = 1.2  # 1 meter
    # clipping_distance = clipping_distance_in_meters / depth_scale
    #
    # align_to = rs.stream.color
    # align = rs.align(align_to)
    # t = 1
    # # # 相机刚开始工作的时候色温不对,因此需要进行预热
    # while True:
    #
    #     frames = pipeline.wait_for_frames()
    #     aligned_frames = align.process(frames)
    #     aligned_depth_frame = aligned_frames.get_depth_frame()
    #     # aligned_depth_frame is a 640x480 depth image
    #
    #     color_frame = frames.get_color_frame()
    #
    #     # Convert images to numpy arrays
    #     depth_image = np.asanyarray(aligned_depth_frame.get_data())
    #     color_image = np.asanyarray(color_frame.get_data())
    #     # Remove background - Set pixels further than clipping_distance to grey
    #     grey_color = 153
    #
    #     # Render images
    #     depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    #
    #     if t == 100:
    #         cv2.imwrite('img.jpg', color_image)
    #         cv2.imwrite('d.jpg', depth_colormap)
    #         # images_stack = np.hstack((bg_removed, depth_colormap))
    #         break
    #     t = t + 1
    #
    # pipeline.stop()
    depth_colormap = cv2.imread('d.jpg')
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    # PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
    PATH_TO_CKPT = 'frozen_inference_graph.pb'
    # List of the strings that is used to add correct label for each box.
    # PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
    PATH_TO_LABELS = 'label.pbtxt'
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

    name = 'item313.jpg'
    image_path = name
    image = Image.open(image_path)
    (im_width, im_height) = image.size
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    # image_np = cv2.resize(image_np,(960,540 ))
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
    cv2.imwrite('detect_result.jpg', img1)
    cv2.imshow('img', img1)
    box = []

    image = cv2.imread(name)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    who_edge = cv2.Canny(img, 20, 150, apertureSize=3)
    (im_height, im_width, chanel) = image.shape
    fh = 550
    fw = 550
    cv2.imshow('Canny', who_edge)
    # TODO: 需要找到桌子的四个定点,然后输入到下面的程序进行透视变换
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = corner.getCorner(name)
    print(y1,y2)
    for k in range(y1,im_height):
        for j in range(im_width):
            who_edge[k][j]=0

    # print(x1,y1,x2,y2,x3,y3,x4,y4)
    pts3 = np.float32([[x3, y3], [x4, y4], [x1, y1], [x2, y2]])
    # pts3 = np.float32([[287, 180], [619, 172], [176, 442], [729, 425]])
    pts4 = np.float32([[0, 0], [fh, 0], [0, fh], [fh, fh]])
    M_perspective = cv2.getPerspectiveTransform(pts3, pts4)
    show_image = cv2.warpPerspective(image, M_perspective, (fh, fh))

    # 新建一个图层来进行线段的绘画,使幕板图为灰度图
    canvas = np.zeros((im_height, im_width, 1), dtype="uint8")

    count = 0
    for i in range(output_dict['detection_boxes'].shape[0]):
        if output_dict['detection_scores'] is None or output_dict['detection_scores'][i] > 0.7:
            print(category_index[output_dict['detection_classes'][i]]['name'],
                  output_dict['detection_scores'][i] and output_dict['detection_scores'][i] != 1.0)
            print(output_dict['detection_boxes'][i])
            a = tuple(output_dict['detection_boxes'][i].tolist())
            count = count + 1
            box.append(a)
    print('Detect number', count)

    for i in range(output_dict['detection_boxes'].shape[0]):
        if output_dict['detection_scores'] is None or output_dict['detection_scores'][i] > 0.7 or (
                output_dict['detection_scores'][i] > 0.5 and (
                category_index[output_dict['detection_classes'][i]]['name'] == 'nn' or category_index[output_dict['detection_classes'][i]]['name'] =='xyy')) and \
                output_dict['detection_scores'][i] != 1.0:
            item = category_index[output_dict['detection_classes'][i]]['name']
            # box = tuple(output_dict['detection_boxes'][i].tolist())
            print(item, output_dict['detection_scores'][i])
            print(i)
            minx = (int)(output_dict['detection_boxes'][i][0] * im_height)
            maxx = (int)(output_dict['detection_boxes'][i][2] * im_height*1.025)
            miny = (int)(output_dict['detection_boxes'][i][1] * im_width)
            maxy = (int)(output_dict['detection_boxes'][i][3] * im_width)
            cv2.rectangle(depth_colormap, (miny, minx), (maxy, maxx), (0, 0, 255), 2)
            crop_list = judge_box(box, i, count)

            if len(crop_list) > 0:
                for j in range(len(crop_list)):
                    index = int(crop_list[j])
                    print(box[index])
                    x1 = output_dict['detection_boxes'][i][0]
                    y1 = output_dict['detection_boxes'][i][1]
                    x2 = output_dict['detection_boxes'][i][2]
                    y2 = output_dict['detection_boxes'][i][3]
                    print(y2, box[index][1], y1)
                    if box[index][2] > x2:
                        if item != 'gz' and item != 'xyy' and item != 'yz':
                            if y2 > box[index][1] > y1:
                                maxy = (int)(box[index][1] * im_width)
                                print('maxy', maxy)
                            elif y1 < box[index][3] < y2:
                                miny = (int)(box[index][3] * im_width)
                                print('miny', miny)


            if miny > maxy:
                maxy = (int)(output_dict['detection_boxes'][i][3] * im_width)
                miny = (int)(output_dict['detection_boxes'][i][1] * im_width)
            # 对图片进行裁剪,得到需要的物体区域框,然后进行滤波和边缘检测
            crop_image = image[minx:maxx, miny:maxy]
            # 获取物体的最佳canny阈值
            lowCanny, highCanny = getCanny(getid(item))
            itemhsv = cv2.cvtColor(cv2.GaussianBlur(image[minx:maxx, miny:maxy], (3, 3), 0), cv2.COLOR_BGR2HSV)
            hsvlist = ['whj', 'kc', 'w', 'bg', 'yz']
            cannylist = ['kxt', 'jd']
            if item in hsvlist:
                if item == 'w':
                    edges = cv2.inRange(itemhsv, (0, 0, 0), (180, 103, 255))
                elif item == 'wjh':
                    edges = cv2.inRange(itemhsv, (0, 0, 0), (180, 43, 255))
                elif item == 'kc':
                    edges = cv2.inRange(itemhsv, (0, 0, 0), (180, 30, 255))
                elif item == 'bg':
                    edges = cv2.inRange(itemhsv, (0, 0, 0), (180, 115, 255))
                elif item == 'yz':
                    edges = cv2.inRange(itemhsv, (0, 0, 0), (180, 104, 255))
                edges = cv2.Canny(edges, 20, 150, apertureSize=3)
            elif item in cannylist:
                if item == 'kxt':
                    lowCanny = 5
                    highCanny = 70
                elif item == 'jd':
                    lowCanny = 10
                    highCanny = 50
                    crop_image = cv2.GaussianBlur(crop_image, (3, 3), 0)
                print('in canny list')
                edges = cv2.Canny(crop_image, lowCanny, highCanny, apertureSize=3)
            else:
                print('in this part')
                cv2.imwrite('a.jpg', crop_image)
                edges = who_edge[minx:maxx, miny:maxy]
            cv2.imshow('cannyresult', edges)
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
            print('两个底部点的y坐标差为', rby - lby)
            print(lbx, lby)
            print(rbx, rby)
            # cv2.imshow('asd',edges)
            # cv2.waitKey()
            if ifsquare(item) == 2:
                pad = 0
                # 　如果两个点分别在框中心的两侧
                if lby < 0.1 * (maxy - miny) and rby > 0.9 * (maxy - miny):
                    lbflag = rbflag = 0
                    print('近似平行')
                    for i in reversed(range(h - 1)):
                        if pad == 0:
                            for j in reversed(range(w)):
                                if (edges[i][j] == 255):
                                    print(maxy + miny, im_width)

                                    # 如果整个物体位于相机的右边
                                    if (maxy + miny) > im_width:
                                        for k in range(w):
                                            if (edges[i - 1][k] == 255):
                                                myx = minx + i
                                                myy = miny + k
                                                lf = 0
                                                rf = round((maxy - miny - lby) * 0.7)
                                                lbflag = 1
                                                for i in reversed(range(h)):
                                                    if (edges[i][rby - lf] == 255):
                                                        print(i)
                                                        temp = i
                                                        break

                                                for i in reversed(range(h)):
                                                    if (edges[i][rby + rf] == 255):
                                                        print(i)
                                                        temp1 = i
                                                        break
                                                break

                                    else:
                                        for k in reversed(range(w)):
                                            if edges[i - 1][k] == 255:
                                                myx = minx + i
                                                myy = miny + k
                                                lf = round(0.7 * rby)
                                                rf = 0
                                                rbflag = 1
                                                for i in reversed(range(h)):
                                                    if (edges[i][rby - lf] == 255):
                                                        print(i)
                                                        temp = i
                                                        break

                                                for i in reversed(range(h)):
                                                    if (edges[i][rby + rf] == 255):
                                                        print(i)
                                                        temp1 = i
                                                        break
                                                break
                                    pad = 1
                                    break
                        else:
                            break
                    print('向左查找距离为', lf, '向右查找的距离为', rf)

                else:
                    # 相对于原图的坐标
                    print('最底部点相对于原图的坐标为', minx + lbx, miny + lby)
                    # 默认情况为按照如下比例进行计算
                    lf = round(0.4 * (lby))
                    rf = round(0.4 * (w - rby))
                    # 计算最底下点也是默认从左往右第一个点
                    # 如果找到的两个点的y相差不大,说明底部还是存在的
                    # if rby-lby<12:
                    print('最底部点位于裁剪图的', (lby) / (maxy - miny))
                    # 重新设置往左和往右的长度和比例
                    rbflag = lbflag = 0

                    myf = 0
                    if item == 'wjh':
                        if (maxy - miny) < (maxx - minx):
                            myf = 0.4
                        else:
                            myf = 0.7
                    else:
                        myf = 0.70

                    if lby > 0.5 * (maxy - miny):
                        lf = round(myf * lby)
                        rf = 0
                        rbflag = 1
                    elif lby <= 0.5 * (maxy - miny):
                        lf = 0
                        rf = round((maxy - miny - lby) * myf)
                        lbflag = 1
                    print('向左查找距离为', lf, '向右查找的距离为', rf)
                    # cv2.imshow('edges', edges)
                    # cv2.imshow('crop', crop_image)
                    # # cv2.imwrite('out.jpg', image)
                    # # 从最下的点开始往左搜索固定长度,如果上方有白点,则进行连线,此处即为底部轮廓的一条边
                    cv2.waitKey()
                    if lby > 0.5 * (maxy - miny) :
                        for i in reversed(range(lbx)):
                            if (edges[i][lby - lf] == 255):
                                print(i)
                                temp = i
                                break

                        for i in reversed(range(lbx)):
                            if (edges[i][lby + rf] == 255):
                                print(i)
                                temp1 = i
                                break
                    else:
                        for i in reversed(range(lbx)):
                            if (edges[i][lby - lf] == 255):
                                print(i)
                                temp = i
                                break

                        for i in reversed(range(lbx)):
                            if (edges[i][lby + rf] == 255):
                                print(i)
                                temp1 = i
                                break

                    if lby > 0.5 * (maxy - miny):
                        myx = minx + rbx
                        myy = miny + rby
                    else:
                        myx = minx + lbx
                        myy = miny + lby
                cv2.imshow('waet', edges)

                print(myx)
                print(myy)

                canvas = np.zeros((im_height, im_width, 1), dtype="uint8")

                if pad == 1:
                    if (maxy + miny) > im_width:
                        cv2.line(image, (myy, myx), (myy + rf, minx + temp1), (0, 255, 255))
                        cv2.line(canvas, (myy, myx), (myy + rf, minx + temp1), 255)
                    else:
                        # print('asdas',myy - lf, minx + temp)
                        cv2.line(image, (myy, myx), (myy - lf, minx + temp), (0, 255, 255))
                        cv2.line(canvas, (myy, myx), (myy - lf, minx + temp), 255)
                        # cv2.imshow('aadsasd',canvas)
                # 在原图上画线段,
                else:
                    if lby > 0.5 * (maxy - miny):
                        cv2.line(image, (myy, myx), (myy - lf, minx + temp), (0, 255, 255))
                        cv2.line(canvas, (myy, myx), (myy - lf, minx + temp), 255)
                    else:
                        cv2.line(image, (myy, myx), (myy + rf, minx + temp1), (0, 255, 255))
                        cv2.line(canvas, (myy, myx), (myy + rf, minx + temp1), 255)
                    # cv2.imshow('aadsasd', canvas)
                # cv2.imshow('aadsasd', canvas)
                # cv2.circle(image, ((myx), (myy)), 1, (0, 255, 255),1)

                # cv2.line(canvas, (miny + lby, minx + lbx), (miny + lby - 40, minx + temp), 255)
                # cv2.line(canvas, (miny + rby, minx + rbx), (miny + rby + 20, minx + temp1), 255)
                # cv2.circle(canvas, ((myx), (myy)), 1, 255)

                # 原图的透视变换
                img_perspective = cv2.warpPerspective(image, M_perspective, (fh, fh))
                f_edges = cv2.Canny(img_perspective, 50, 150, apertureSize=3)
                # cv2.imshow('f_edge', f_edges)

                # after_perspective = cv2.inRange(img_perspective, (54, 254, 154), (0, 255, 255))
                # cv2.imshow('after_perspective', after_perspective)

                # 幕板的透视变换,变换后缩放到桌子的长度,则图上1个像素点表示1mm

                perspective = cv2.warpPerspective(canvas, M_perspective, (fh, fh))

                # 由于变换后的直线出现了形变,因此筛选处一些点
                for i in range(fh):
                    for j in range(fw):
                        if perspective[i][j] < 150:
                            perspective[i][j] = 0
                cv2.imshow('ee', perspective)
                cv2.waitKey()
                # 寻找最底部的点.默认从左往右,但是如果最底部的点靠右,则从右往左查找
                if rbflag == 1:
                    flag = 0
                    for i in reversed(range(fh)):
                        if flag == 0:
                            for j in reversed(range(fw)):
                                if (perspective[i][j] != 0):
                                    per_bx = i
                                    per_by = j
                                    flag = 1
                                    break
                        else:
                            break

                if lbflag == 1:
                    flag = 0
                    for i in reversed(range(fh)):
                        if flag == 0:
                            for j in range(fw):
                                if (perspective[i][j] != 0):
                                    per_bx = i
                                    per_by = j
                                    flag = 1
                                    break
                        else:
                            break
                # 寻找左边的线段点
                mminy = 1000
                for i in range(fh):
                    for j in range(fw):
                        if (perspective[i][j] != 0 and j < mminy):
                            mminy = j
                            flag = 1
                            mminx = i

                # 寻找右边的线段点
                mmaxy = 0
                for i in range(fh):
                    for j in reversed(range(fw)):
                        if (perspective[i][j] != 0 and j > mmaxy):
                            mmaxy = j
                            flag = 1
                            mmaxx = i

                print("最底部点坐标为")
                print(per_bx, per_by)
                cv2.imshow('ruuwer', perspective)
                print("最左边坐标为", mminx, mminy)
                print("最右边坐标为", mmaxx, mmaxy)

                # 角度和长度的计算
                b1 = ((mminy - per_by) * (mminy - per_by) + (mminx - per_bx) * (mminx - per_bx)) ** 0.5
                b2 = ((mmaxy - per_by) * (mmaxy - per_by) + (mmaxx - per_bx) * (mmaxx - per_bx)) ** 0.5
                print("线段长为", b1, b2)
                # itemchang,itemkuan = getitem(getid(category_index[output_dict['detection_classes'][i]]['name']))
                print(miny, maxy)

                if pad == 1:
                    if rbflag == 1:
                        itemright, itemleft = (getitem(getid(item)))
                        newmmaxy = per_by + itemright * ((per_bx - mminx) / b1)
                        newmmaxx = per_bx - itemright * ((per_by - mminy) / b1)
                        newmminy = per_by - itemleft * (per_by - mminy) / b1
                        newmminx = per_bx - itemleft * (per_bx - mminx) / b1
                        cv2.line(perspective, (per_by, per_bx), (round(newmmaxy), round(newmmaxx)), 255)
                        cv2.line(show_image, (per_by, per_bx), (round(newmmaxy), round(newmmaxx)), 255)
                        cv2.line(show_image, (per_by, per_bx), (round(newmminy), round(newmminx)), 255)
                        cv2.line(img_perspective, (per_by, per_bx), (round(newmmaxy), round(newmmaxx)), 255)
                        theta = math.asin((per_bx - mminx) / b1) * 180 / math.pi
                        print('pad=1,点右长边右', newmmaxx, newmmaxy)
                    elif lbflag == 1:
                        itemleft, itemright = (getitem(getid(item)))
                        newmmaxy = per_by - itemright * ((per_by - mmaxy) / b2)
                        newmmaxx = per_bx - itemright * ((per_bx - mmaxx) / b2)
                        newmminy = per_by - itemleft * (per_bx - mmaxx) / b2
                        newmminx = per_bx - itemleft * (mmaxy - per_by) / b2
                        cv2.line(perspective, (per_by, per_bx), (round(newmminy), round(newmminx)), 255)
                        cv2.line(show_image, (per_by, per_bx), (round(newmminy), round(newmminx)), 255)
                        cv2.line(show_image, (per_by, per_bx), (round(newmmaxy), round(newmmaxx)), 255)
                        cv2.line(img_perspective, (per_by, per_bx), (round(newmminy), round(newmminx)), 255)
                        theta = math.asin((mmaxy - per_by) / b2) * 180 / math.pi
                        print('pad=1,点左长边左', newmmaxx, newmmaxy)
                else:
                    if lby > 0.5 * (maxy - miny):
                        itemleft, itemright = (getitem(getid(item)))
                        if itemright > b1:
                            itemright, itemleft = (getitem(getid(item)))
                            newmmaxy = per_by + itemright * ((per_bx - mminx) / b1)
                            newmmaxx = per_bx - itemright * ((per_by - mminy) / b1)
                            newmminy = per_by - itemleft * (per_by - mminy) / b1
                            newmminx = per_bx - itemleft * (per_bx - mminx) / b1
                            cv2.line(perspective, (per_by, per_bx), (round(newmmaxy), round(newmmaxx)), 255)
                            cv2.line(show_image, (per_by, per_bx), (round(newmmaxy), round(newmmaxx)), 255)
                            cv2.line(show_image, (per_by, per_bx), (round(newmminy), round(newmminx)), 255)
                            cv2.line(img_perspective, (per_by, per_bx), (round(newmmaxy), round(newmmaxx)), 255)
                            theta = math.asin((per_by - mminy) / b1) * 180 / math.pi
                            print('点右长边右', newmmaxx, newmmaxy)
                        else:
                            itemleft, itemright = (getitem(getid(item)))
                            newmmaxy = per_by + itemright * ((per_bx - mminx) / b1)
                            newmmaxx = per_bx - itemright * ((per_by - mminy) / b1)
                            newmminy = per_by - itemleft * (per_by - mminy) / b1
                            newmminx = per_bx - itemleft * (per_bx - mminx) / b1
                            cv2.line(perspective, (per_by, per_bx), (round(newmmaxy), round(newmmaxx)), 255)
                            cv2.line(show_image, (per_by, per_bx), (round(newmmaxy), round(newmmaxx)), 255)
                            cv2.line(show_image, (per_by, per_bx), (round(newmminy), round(newmminx)), 255)
                            cv2.line(img_perspective, (per_by, per_bx), (round(newmmaxy), round(newmmaxx)), 255)
                            theta = math.asin((per_bx - mminx) / b1) * 180 / math.pi
                            print('点右长边左', newmmaxx, newmmaxy)
                    else:
                        itemright, itemleft = (getitem(getid(item)))
                        if itemleft > b2:
                            itemleft, itemright = (getitem(getid(item)))
                            newmmaxy = per_by - itemright * ((per_by - mmaxy) / b2)
                            newmmaxx = per_bx - itemright * ((per_bx - mmaxx) / b2)
                            newmminy = per_by - itemleft * (per_bx - mmaxx) / b2
                            newmminx = per_bx - itemleft * (mmaxy - per_by) / b2
                            cv2.line(perspective, (per_by, per_bx), (round(newmminy), round(newmminx)), 255)
                            cv2.line(show_image, (per_by, per_bx), (round(newmminy), round(newmminx)), 255)
                            cv2.line(show_image, (per_by, per_bx), (round(newmmaxy), round(newmmaxx)), 255)
                            cv2.line(img_perspective, (per_by, per_bx), (round(newmminy), round(newmminx)), 255)
                            theta = math.asin((mmaxy - per_by) / b2) * 180 / math.pi
                            print('点左长边左', newmmaxx, newmmaxy)
                        else:
                            itemright, itemleft = (getitem(getid(item)))
                            newmmaxy = per_by - itemright * ((per_by - mmaxy) / b2)
                            newmmaxx = per_bx - itemright * ((per_bx - mmaxx) / b2)

                            newmminy = per_by - itemleft * (per_bx - mmaxx) / b2
                            newmminx = per_bx - itemleft * (mmaxy - per_by) / b2
                            cv2.line(perspective, (per_by, per_bx), (round(newmminy), round(newmminx)), 255)
                            cv2.line(show_image, (per_by, per_bx), (round(newmminy), round(newmminx)), 255)
                            cv2.line(show_image, (per_by, per_bx), (round(newmmaxy), round(newmmaxx)), 255)
                            cv2.line(img_perspective, (per_by, per_bx), (round(newmminy), round(newmminx)), 255)
                            theta = math.asin((mmaxx - per_bx) / b2) * 180 / math.pi
                            print('点左长边右', newmminx, newmminy)

                # # 通常情况下按照三个点和尺寸计算得到的坐标位置信息
                # x3 = per_bx - (itemleft * (per_bx - mminx) / b1)
                # y3 = per_by - (itemleft * (per_by - mminy) / b1)
                # x4 = per_bx + (itemright * (mmaxx - per_bx) / b2)
                # y4 = per_by - (itemright * (per_by - mmaxy) / b2)

                print("newmminx坐标为", newmminx, newmminy)
                print("newmmaxx坐标为", newmmaxx, newmmaxy)
                print('物体为', item, '编号为', getid(item))
                print("中心坐标为", (newmminx + newmmaxx) / 2, (newmminy + newmmaxy) / 2)

                print("角度为", theta)
                itemId.append(getid(item) + '    ')
                positionx.append(str(round((newmminy + newmmaxy) / 2)) + ' ')
                positiony.append(str(round((newmminx + newmmaxx) / 2)) + ' ')
                Theta.append(str(round(theta, 1)) + '  ')

                # cv2.imshow('output', canvas)
                # cv2.imshow('edges', edges)
                # cv2.imshow('contour', show_image)

            elif ifsquare(item) == 1:
                # 直接取中点来计算
                nx = round((lbx + rbx) / 2) + minx
                ny = round((lby + rby) / 2) + miny
                canvas = np.zeros((im_height, im_width, 1), dtype="uint8")
                cv2.circle(canvas, (ny, nx), 1, 255)
                cv2.circle(image, (ny, nx), 1, (255, 255, 0))
                cv2.circle(image, (lby, lbx), 1, (255, 255, 0))
                cv2.circle(image, (rby, rbx), 1, (255, 255, 0))
                img_perspective = cv2.warpPerspective(image, M_perspective, (fh, fh))
                cv2.imshow('addv', img_perspective)
                cv2.waitKey()

                # img_perspective = cv2.GaussianBlur(img_perspective,(3,3),0)
                f_edges = cv2.Canny(img_perspective, 0, 150, apertureSize=3)
                # cv2.imshow('edges', edges)
                # cv2.imshow('crop', crop_image)
                perspective = cv2.warpPerspective(canvas, M_perspective, (fh, fh))
                # cv2.imshow('rs', perspective)

                # 筛选掉一些点
                for i in range(fh):
                    for j in range(fw):
                        if perspective[i][j] < 150:
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
                # cv2.imshow('per',perspective)
                # cv2.waitKey()
                print(bx, by)
                bx = bx - getitem(getid(item)) / 2
                print(item, '的中心位置为', bx, by)
                itemId.append(getid(item) + '     ')
                positionx.append(str(round(by)) + ' ')
                positiony.append(str(round(bx)) + ' ')
                Theta.append('None' + ' ')
            elif ifsquare(item) == 0:
                cv2.imshow('crop', crop_image)
                cv2.waitKey()
                by = (maxy + miny) / 2
                bx = maxx / 1
                img_perspective = cv2.warpPerspective(image, M_perspective, (fh, fh))
                canvas = np.zeros((im_height, im_width, 1), dtype="uint8")
                cv2.circle(canvas, (round(by), round(bx)), 1, 255)
                perspective = cv2.warpPerspective(canvas, M_perspective, (fh, fh))

                flag = 0
                for i in (range(w)):
                    if flag == 0:
                        for j in range(h):
                            if (edges[j][i] == 255):
                                lby = i
                                lbx = j
                                flag = 1
                                print(lbx, lby)
                                break
                    else:
                        break
                flag = 0
                for i in reversed(range(w)):
                    if flag == 0:
                        for j in (range(h)):
                            if (edges[j][i] == 255):
                                rby = i
                                rbx = j
                                flag = 1
                                print(rbx, rby)
                                break
                    else:
                        break

                c = (rbx - lby)
                iteml, itemr = getitem(getid(item))
                ph = math.atan(itemr / iteml)
                th = math.asin(c / ((iteml * iteml + itemr * itemr) ** 0.5)) * 180 / math.pi - ph
                print(th)
                # # 筛选掉一些点
                # for i in range(fh):
                #     for j in range(fw):
                #         if perspective[i][j] < 150:
                #             perspective[i][j] = 0

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

                if (maxx - minx) > (maxy - miny):
                    bx = bx - iteml / 2
                else:
                    bx = bx - itemr / 2
                itemId.append(getid(item) + '     ')
                positionx.append(str(round(by)) + ' ')
                positiony.append(str(round(bx)) + ' ')
                Theta.append(str(round(-th, 1)) + ' ')

    # plt.figure(figsize=IMAGE_SIZE)
    # plt.imshow(image_np)

    img1 = cv2.imread('detect_result.jpg')

    who_edge = cv2.cvtColor(who_edge, cv2.COLOR_GRAY2BGR)
    print(img1.shape, who_edge.shape)
    depth_colormap = cv2.resize(depth_colormap, (960, 540))
    images_stack = np.hstack((img1, who_edge, depth_colormap))
    images_stack = cv2.resize(images_stack, (1920, 360))
    cv2.imshow('Result1', images_stack)
    cv2.moveWindow('hello', 0, 0)
    show_image = cv2.resize(show_image, (550, 550))
    cv2.imshow('contour', show_image)
    cv2.moveWindow('contour', 0, 500)
    cv2.imwrite('contour.jpg',show_image)
    # cv2.imshow('contour', show_image)
    # cv2.imshow('object',img1)
    # cv2.imshow('canny',who_edge)
