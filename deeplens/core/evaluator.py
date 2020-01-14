import os
import re
import time
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from core import build_depth_resnet, build_lensblur, build_sr


def restore_ete_model(sess, variables_to_restore, eval_config):
    saver = tf.train.Saver(variables_to_restore)
    saver.restore(sess=sess, save_path=eval_config.ete_ckpt)


def restore_piecewise_model(sess, pre_trained_params, eval_config):
    name_to_var_depth = dict([('Network' + re.split('Depth', v.op.name)[-1], v) for v in pre_trained_params if
                              re.search('Depth', v.op.name)])
    name_to_var_lensblur = dict([(v.op.name, v) for v in pre_trained_params if
                                 re.search('Lensblur', v.op.name) or re.search('Feature', v.op.name) or re.search('SR',
                                                                                                                  v.op.name)])
    init_saver_depth = tf.train.Saver(name_to_var_depth)
    init_saver_lensblur = tf.train.Saver(name_to_var_lensblur)
    init_saver_depth.restore(sess, eval_config.depth_ckpt)
    init_saver_lensblur.restore(sess, eval_config.lensblur_ckpt)


def construct_graph(eval_config):
    image_320_input_tensor = tf.placeholder(dtype=tf.float32, name='image_320', shape=[None, None, 3])
    image_640_input_tensor = tf.placeholder(dtype=tf.float32, name='image_640', shape=[None, None, 3])
    image_1280_input_tensor = tf.placeholder(dtype=tf.float32, name='image_1280', shape=[None, None, 3])
    image_320_tensor = tf.expand_dims(image_320_input_tensor, 0)
    image_640_tensor = tf.expand_dims(image_640_input_tensor, 0)
    image_1280_tensor = tf.expand_dims(image_1280_input_tensor, 0)
    aperture_tensor = tf.placeholder(tf.float32, name='aperture', shape=[])
    focal_x_tensor = tf.placeholder(tf.int32, name='focal_x', shape=[])
    focal_y_tensor = tf.placeholder(tf.int32, name='focal_y', shape=[])
    is_training = tf.constant(False, dtype=tf.bool, shape=[])

    with tf.variable_scope('Network'):
        depth_320, depth_net = build_depth_resnet.build(image_320_tensor, is_training)
        focal_depth = depth_320[0, focal_y_tensor, focal_x_tensor, 0]
        depth_320_signed = (depth_320 - focal_depth) * aperture_tensor
        pre_dof_320, lensblur_net, feature_net = build_lensblur.build(image_320_tensor, depth_320_signed, is_training)
        pre_dof_640, sr_net = build_sr.build(image_320_tensor, depth_320_signed, pre_dof_320, image_640_tensor,
                                             is_training)
        shape_640 = tf.shape(pre_dof_640)
        depth_640_signed = tf.image.resize_bilinear(depth_320_signed, [shape_640[1], shape_640[2]])
        pre_dof_1280, sr_net = build_sr.build(image_640_tensor, depth_640_signed, pre_dof_640, image_1280_tensor,
                                              is_training)

    variables_to_restore = tf.global_variables()

    session_config = tf.ConfigProto()
    # faster if smaller?
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.8
    sess = tf.Session(config=session_config)

    # restore net params
    if eval_config.use_moving_average:
        ema = tf.train.ExponentialMovingAverage(0.9)
        variables_to_restore = ema.variables_to_restore()

    if eval_config.HasField('depth_ckpt') and eval_config.HasField('lensblur_ckpt'):
        restore_piecewise_model(sess, variables_to_restore, eval_config)
    elif eval_config.HasField('ete_ckpt'):
        restore_ete_model(sess, variables_to_restore, eval_config)
    else:
        raise ValueError("No pre-trained models")

    def pre_dof(im):
        focal_x, focal_y = im.get_focal_point()
        aperture = im.aperture
        if im.scale == 4:
            return sess.run(pre_dof_1280,
                            feed_dict={image_320_input_tensor: im.im_320, image_640_input_tensor: im.im_640,
                                       image_1280_input_tensor: im.im_1280, focal_x_tensor: focal_x,
                                       focal_y_tensor: focal_y, aperture_tensor: aperture})
        elif im.scale == 2:
            return sess.run(pre_dof_640,
                            feed_dict={image_320_input_tensor: im.im_320, image_640_input_tensor: im.im_640,
                                       focal_x_tensor: focal_x, focal_y_tensor: focal_y, aperture_tensor: aperture})
        elif im.scale == 1:
            return sess.run(pre_dof_320, feed_dict={image_320_input_tensor: im.im_320, focal_x_tensor: focal_x,
                                                    focal_y_tensor: focal_y, aperture_tensor: aperture})
        # return im.im

    def pre_depth(im_320):
        return sess.run(depth_320, feed_dict={image_320_input_tensor: im_320})

    return pre_dof, pre_depth


def evaluate(eval_config):
    run_dof_func, run_depth_func = construct_graph(eval_config)
    # image_path = '/media/4TB/Research/DataSet/Wild/Image/'
    image_path = eval_config.image_path

    res_path = eval_config.res_path + 'dof/'
    res_depth_path = eval_config.res_path + 'depth/'
    if not os.path.isdir(eval_config.res_path):
        os.mkdir(eval_config.res_path)
    if not os.path.isdir(res_path):
        os.mkdir(res_path)
    if not os.path.isdir(res_depth_path):
        os.mkdir(res_depth_path)

    image_names = os.listdir(image_path)
    for img_id, image_name in enumerate(image_names):
        # if img_id % 10 == 0:
        if not image_name.startswith('.'):
            print("Processing Img: " + image_name + "\t%d/%d" % (img_id, len(image_names)))

            im = cv2.imread(image_path + image_name)
            height, width, channels = im.shape
            print("Img Size:\t%d*%d" % (height, width))
            im = im.astype(np.float32) / 255.0

            # im = cv2.resize(im, (320, 320), interpolation=cv2.INTER_LINEAR)
            im = ImageWrapper(im, x=160, y=160)
            depth = run_depth_func(im.im_320)
            im.depth = depth[0, :, :, 0]
            depth = cv2.applyColorMap(np.uint8(depth[0, :, :, 0] * 255), cv2.COLORMAP_JET)

            # print(depth)
            cv2.imwrite(res_depth_path + image_name, depth)
            start_time = time.time()
            dof = run_dof_func(im)
            dof = np.clip(dof[0], 0, 1) * 255
            # diff = im.im * 255 - dof
            # diff = diff * diff
            end_time = time.time()
            print('Spend time:\t%fs' % (end_time - start_time))
            cv2.imwrite(res_path + image_name, dof)


class ImageWrapper(object):
    def __init__(self, im, depth=None, dof=None, x=0, y=0, aperture=1.0):
        if np.max(im.shape) > 1280:
            scale = 1280.0 / np.max(im.shape)
            shape = np.int32(np.array(im.shape) * scale)
            im = cv2.resize(im, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
        self.im = im
        if np.max(im.shape) >= 1280:
            self.scale = 4
            self.im_1280 = im
            self.im_640 = cv2.resize(self.im_1280, (self.im_1280.shape[1] // 2, self.im_1280.shape[0] // 2),
                                     interpolation=cv2.INTER_AREA)
            self.im_320 = cv2.resize(self.im_640, (self.im_640.shape[1] // 2, self.im_640.shape[0] // 2),
                                     interpolation=cv2.INTER_AREA)
        elif np.max(im.shape) >= 640:
            self.scale = 2
            self.im_1280 = None
            self.im_640 = im
            self.im_320 = cv2.resize(self.im_640, (self.im_640.shape[1] // 2, self.im_640.shape[0] // 2),
                                     interpolation=cv2.INTER_AREA)
        else:
            self.scale = 1
            self.im_1280 = None
            self.im_640 = None
            self.im_320 = im

        self.x = x
        self.y = y
        self.depth = depth
        self.dof_320 = dof
        self.aperture = aperture

    def set_focal_point(self, x, y):
        self.x = np.int32(x)  # * self.scale
        self.y = np.int32(y)  # * self.scale

    def get_focal_point(self):
        return self.x / self.scale, self.y / self.scale

    def get_focal_depth(self):
        x, y = self.get_focal_point()
        return self.depth[y, x]

    def set_dof(self, dof):
        self.dof_320 = cv2.resize(dof, (self.im_320.shape[1], self.im_320.shape[0]), interpolation=cv2.INTER_AREA)
