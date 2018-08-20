import numpy as np
import os
import pyrealsense2 as rs
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import math
import corner

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
    PATH_TO_CKPT = '/home/yingkai/Documents/Models/Object_detection_Competition/frozen_inference_graph.pb'
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

    name = '/home/yingkai/Documents/Data/8.9/test1.jpg'
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
