
# coding: utf-8

# In[1]:

import tensorflow as tf
import os
import random
import math
import sys
from PIL import Image
import numpy as np


# In[2]:

#验证集数量
_NUM_TEST = 500

#随机种子
_RANDOM_SEED = 0

#数据集路径
DATASET_DIR = "D:/Tensorflow/captcha/images/"

#tfrecord文件存放路径
TFRECORD_DIR = "D:/Tensorflow/captcha/"


#判断tfrecord文件是否存在
def _dataset_exists(dataset_dir):
    for split_name in ['train', 'test']:
        output_filename = os.path.join(dataset_dir,split_name + '.tfrecords')
        if not tf.gfile.Exists(output_filename):
            return False
    return True

#获取所有验证码图片
def _get_filenames_and_classes(dataset_dir):
    photo_filenames = []
    for filename in os.listdir(dataset_dir):
        #获取文件路径
        path = os.path.join(dataset_dir, filename)
        photo_filenames.append(path)
    return photo_filenames

def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def image_to_tfexample(image_data, label0, label1, label2, label3):
    #Abstract base class for protocol messages.
    return tf.train.Example(features=tf.train.Features(feature={
      'image': bytes_feature(image_data),
      'label0': int64_feature(label0),
      'label1': int64_feature(label1),
      'label2': int64_feature(label2),
      'label3': int64_feature(label3),
    }))

#把数据转为TFRecord格式
def _convert_dataset(split_name, filenames, dataset_dir):
    assert split_name in ['train', 'test']

    with tf.Session() as sess:
        #定义tfrecord文件的路径+名字
        output_filename = os.path.join(TFRECORD_DIR,split_name + '.tfrecords')
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            for i,filename in enumerate(filenames):
                try:
                    sys.stdout.write('\r>> Converting image %d/%d' % (i+1, len(filenames)))
                    sys.stdout.flush()

                    #读取图片
                    image_data = Image.open(filename)  
                    #根据模型的结构resize
                    image_data = image_data.resize((224, 224))
                    #灰度化
                    image_data = np.array(image_data.convert('L'))
                    #将图片转化为bytes
                    image_data = image_data.tobytes()              

                    #获取label
                    labels = filename.split('/')[-1][0:4]
                    num_labels = []
                    for j in range(4):
                        num_labels.append(int(labels[j]))
                                            
                    #生成protocol数据类型
                    example = image_to_tfexample(image_data, num_labels[0], num_labels[1], num_labels[2], num_labels[3])
                    tfrecord_writer.write(example.SerializeToString())
                    
                except IOError as e:
                    print('Could not read:',filename)
                    print('Error:',e)
                    print('Skip it\n')
    sys.stdout.write('\n')
    sys.stdout.flush()

#判断tfrecord文件是否存在
if _dataset_exists(TFRECORD_DIR):
    print('tfcecord文件已存在')
else:
    #获得所有图片
    photo_filenames = _get_filenames_and_classes(DATASET_DIR)

    #把数据切分为训练集和测试集,并打乱
    random.seed(_RANDOM_SEED)
    random.shuffle(photo_filenames)
    training_filenames = photo_filenames[_NUM_TEST:]
    testing_filenames = photo_filenames[:_NUM_TEST]

    #数据转换
    _convert_dataset('train', training_filenames, DATASET_DIR)
    _convert_dataset('test', testing_filenames, DATASET_DIR)
    print('生成tfcecord文件')


# In[ ]:



