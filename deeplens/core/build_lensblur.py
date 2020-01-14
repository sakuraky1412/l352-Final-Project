import tensorflow as tf
from network.kernel_net import KernelNet
from network.feature_net import FeaNet
import numpy as np


def build(im320_tensor, depth320_tensor, is_training):
    with tf.variable_scope('Network'):
        with tf.variable_scope('Lensblur'):
            lens_net = KernelNet({'image': im320_tensor-0.5, 'depth': depth320_tensor}, is_training)
            kernel = lens_net.get_output()
        with tf.variable_scope('Feature'):
            fea_net = FeaNet({'image': im320_tensor}, is_training)
            feature = fea_net.get_output()
    feature = tf.stack([feature[:,:,:,::3], feature[:,:,:,1::3], feature[:,:,:,2::3]], axis=0)
    kernel = kernel / (tf.reduce_sum(kernel, axis=3, keep_dims=True) + np.finfo("float").eps)
    dof_320 = tf.reduce_sum(kernel * feature, axis=4)
    dof_320 = tf.transpose(dof_320, [1,2,3,0])
    return dof_320, lens_net, fea_net
