import os
import tensorflow as tf
import cv2
import random
import numpy as np
import math

test_ratio = 2
crop_size = 224
scale_size = 256
n_classes = 30

def get_records(dataset_path, ext=".jpg"):
    writer_train = tf.python_io.TFRecordWriter("train.tfrecords")
    writer_test = tf.python_io.TFRecordWriter("test.tfrecords")
    train_txt = open("train.txt", "w")
    test_txt = open("test.txt", "w")
    class_names = [f for f in os.listdir(dataset_path) if not f.startswith('.')]
    list.sort(class_names)

    for index, name in enumerate(class_names):
        #print(index, ",", name)
        directory = os.path.join(dataset_path, name)
        class_image_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(ext)]
        
        #print(class_image_paths)
        
        num = len(class_image_paths)
        #### num//5  80%    num//2   50%
        random_sample = random.sample(range(num), num // 2)

        for i, img_path in enumerate(class_image_paths):
            #print(i)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)

            img_raw = img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            # 6 2 2
            if i in random_sample:
                writer_train.write(example.SerializeToString())
                train_txt.write(img_path+'\n')
            else:
                writer_test.write(example.SerializeToString())
                test_txt.write(img_path + '\n')

    writer_train.close()
    writer_test.close()
    train_txt.close()
    test_txt.close()


def mean_image_subtraction(image):
    means = [103.939, 116.779, 123.68] #BGR Mean

    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=2, values=channels)


def read_and_decode(filename, is_train=True):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [256, 256, 3])
    img = tf.cast(img, tf.float32)
    label = tf.cast(features['label'], tf.int32)

    if is_train:
        img = tf.random_crop(img, [224, 224, 3])
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, max_delta=20)
        img = tf.image.rot90(img, tf.random_uniform([], 0, 5, dtype=tf.int32))    

    image_subtracted = mean_image_subtraction(img)
    image_subtracted = image_subtracted[:, :, ::-1]
    return image_subtracted, label


def input_pipeline(filename, batch_size, is_shuffle=True, is_train=True):
    example, label = read_and_decode(filename, is_train)
    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3 * batch_size
    if is_shuffle:
        example_batch, label_batch = tf.train.shuffle_batch([example, label],
                                                            batch_size=batch_size,
                                                            capacity=capacity,
                                                            min_after_dequeue=min_after_dequeue,
                                                            num_threads=4)
    else:
        example_batch, label_batch = tf.train.batch([example, label], batch_size=batch_size, capacity=capacity, num_threads=4)
    return example_batch, label_batch



if __name__ == '__main__':
    
    get_records("AID", ext='.jpg')
    
    
