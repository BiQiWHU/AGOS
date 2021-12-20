
#////#-*- coding:utf-8 -*-

from network_cam import AGOS
from sklearn.externals import joblib
import numpy as np
import tensorflow as tf
import os
import re
import time
from tensorflow.contrib.slim.nets import resnet_v1
import tensorflow.contrib.slim as slim
from train_cam import accuracy_of_batch

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


def get_variables_to_restore():
	checkpoint_exclude_scopes = ['resnet_v1_50/logits/biases']
	exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]
	variables_to_restore = []
	variables_to_exclude = []
	for var in tf.trainable_variables():
		excluded = False
	for exclusion in exclusions:
		if var.op.name.startswith(exclusion):
			excluded = True
			variables_to_exclude.append(var)
		break
	if not excluded:
		variables_to_restore.append(var)
	return variables_to_restore


def get_label_pred(filename):
    f1=filename
    f = "checkpoints/"
    test_tfrecords = 'test.tfrecords'
    batch_size = 20

    img, label = input_pipeline(test_tfrecords, batch_size, is_shuffle=False, is_train=False)

    img_float = tf.to_float(img)
    
    image_subtracted = mean_image_subtraction(img_float)
    image_subtracted = image_subtracted[:, :, :, ::-1]

    # Model
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        logits, end_points = resnet_v1.resnet_v1_50(image_subtracted, num_classes=None, is_training=False, global_pool=False)
    
    fmap1 = end_points['resnet_v1_50/block4/unit_3/bottleneck_v1']	
    instance_pooling, conv6, attention = AGOS(fmap1, is_training=False)


    accuracy = accuracy_of_batch(instance_pooling, label)
    pred = tf.cast(tf.argmax(instance_pooling, 1), tf.int32)


    saver = tf.train.Saver()

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, f + f1)

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
        
		#label_preds = []
		#label_map = []
		#attention_map = []
		#imgs = []
        acc_sum = 0

        start = time.clock()

        for i in range(250):
            acc, label_, pred_, img_, conv6_, attention_ = sess.run([accuracy, label, pred, img, conv6, attention])
			#x = np.hstack([label_[:, np.newaxis], pred_[:, np.newaxis]])
			#label_preds.append(x)
			#label_map.append(conv6_)
			#attention_map.append(attention_)
			#imgs.append(img_)
            acc_sum += acc

        print('mean_acc:', acc_sum / 250)

        elapsed = (time.clock() - start)
        print("Time used:", elapsed)

        coord.request_stop()
        coord.join(threads)

		#tf.reset_default_graph()

        return acc_sum,elapsed


if __name__ == '__main__':
	f = "checkpoints/"

	fs = os.listdir(f)
	fs1 = []
	for f1 in fs:
		(f1name, f1extension) = os.path.splitext(f1)
		fs1.append(f1name)

	fs1 = list(set(fs1))
	fs1 = sorted(fs1, key=lambda i: int(re.search(r'(\d+)', i).group()))
	# print(fs1)

	## creat txt file
	file1 = open("name.txt", 'w')
	file2 = open("acc.txt", 'w')
	file3 = open("time.txt", 'w')

	row = 0

	for f1 in fs1:
		print(f1)

		acc_sum,elapsed=get_label_pred(f1)
		# 
		#with open('name.txt', 'a') as file1:
		#     file1.write(f1 + '\n')

		file1=open('name.txt', 'a') 
		try:
			file1.write(f1 + '\n')
		finally:
			file1.close()
   	
		file2=open('acc.txt', 'a')
		try:
			#file2.write(f1 + '\n')
			#file2.write("{:.2f}%".format(acc_sum  + '\n'))
			file2.write(str(acc_sum)  + '\n')

		finally:
			file2.close()

		file3=open('time.txt', 'a')
		try:
			#file3.write(f1 + '\n')
			file3.write(str(elapsed) + '\n')
		finally:
			file3.close()
		
		# with open('acc.txt', 'a') as file2:
		#     file2.write("{:.2f}%".format(acc_sum * 100 / 10 + '\n')

		# with open('time.txt', 'a') as file3:
		#     file3.write(str(elapsed) + '\n')

		tf.reset_default_graph()

		row = row + 1

		## save txt
	
	file1.close()
	file2.close()
	file3.close()

	print('finish!')
