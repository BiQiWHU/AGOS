from train_cam import accuracy_of_batch2
from train_cam import accuracy_of_batch
from tensorflow.contrib.slim.nets import resnet_v1
from sklearn.externals import joblib
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from network_cam import AGOS
import math

def mean_image_subtraction(image):
    means = [103.939, 116.779, 123.68] #BGR Mean

    if image.get_shape().ndims != 4:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=image)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)


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
    
    label = tf.cast(features['label'], tf.int32)

    return img, label


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
        example_batch, label_batch = tf.train.batch([example, label], batch_size=batch_size, num_threads=1)
    return example_batch, label_batch


def get_label_pred(ckpt_num, is_save=False):
    test_tfrecords = 'test.tfrecords'
    ckpt = 'checkpoints/my-model.ckpt-' + str(ckpt_num)
    batch_size = 20

    img, label = input_pipeline(test_tfrecords, batch_size, is_shuffle=False, is_train=False)
    img_float = tf.to_float(img)
    image_subtracted = mean_image_subtraction(img_float)
    image_subtracted = image_subtracted[:, :, :, ::-1]
    
    # Model
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        instance_pooling, end_points = resnet_v1.resnet_v1_50(image_subtracted, num_classes=None, is_training=False, global_pool=False)
    
    #instance_pooling, conv6, attention = vc_net(image_subtracted, is_training=False)
    #print(tf.shape(instance_pooling))
    #print(tf.shape(label))

    #accuracy = accuracy_of_batch2(instance_pooling, label)
    
    #instance_pooling=tf.squeeze(instance_pooling)
    #pred = tf.cast(tf.argmax(instance_pooling, 1), tf.int32)
    
    #print(tf.shape(instance_pooling))
    #print(tf.shape(label))
    
    fmap1 = end_points['resnet_v1_50/block4/unit_3/bottleneck_v1']	
    instance_pooling, conv6, attention = AGOS(fmap1, is_training=False)


    accuracy = accuracy_of_batch(instance_pooling, label)
    pred = tf.cast(tf.argmax(instance_pooling, 1), tf.int32)
    
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        saver.restore(sess, ckpt)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        total_parameters = 0
        # iterating over all variables
        for variable in tf.trainable_variables():
            local_parameters = 1
            shape = variable.get_shape()  # getting shape of a variable
            for i in shape:
                local_parameters *= i.value  # mutiplying dimension values
            total_parameters += local_parameters
        print(total_parameters)
        
        
        label_preds = []
        label_map = []
        attention_map = []
        imgs = []
        acc_sum = 0
        for i in range(250):
            
            acc, label_, pred_, img_ = sess.run([accuracy, label, pred, img])
            
            #print(label_)
            #print(pred_)
            
            x = np.hstack([label_[:, np.newaxis], pred_[:, np.newaxis]])
            label_preds.append(x)
            #label_map.append(conv6_)
            #attention_map.append(attention_)
            imgs.append(img_)
            acc_sum += acc
        print('mean_acc:', acc_sum / 250)
        label_preds_packed = np.vstack(label_preds)
        if is_save:
            joblib.dump(label_preds_packed, 'label_pred_train')
            joblib.dump(label_map, 'label_map')
            #joblib.dump(attention_map, 'attention_map')
            joblib.dump(imgs, 'imgs')

        coord.request_stop()
        coord.join(threads)
    tf.reset_default_graph()


if __name__ == '__main__':

    get_label_pred(17000, False)