
from network_cam import AGOS
from tfdata import *
import tensorflow.contrib.slim as slim
import os
from tensorflow.contrib.slim.nets import resnet_v1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def bag_ce(logits, targets):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)
    cross_entropy = tf.reduce_mean(cross_entropy, name='cross_entropy')
    return cross_entropy

####

def attention_entropy(attention_weight):
    entropy = tf.reduce_sum(-tf.multiply(attention_weight,tf.log(tf.clip_by_value(attention_weight, 1e-10, 1.0))),[1,2])
    mean_entropy = tf.reduce_mean(entropy)
    return mean_entropy


def attention_reg(attention_weight):
    entropy = tf.reduce_sum(tf.square(attention_weight), [1, 2])
    mean_entropy = tf.reduce_mean(entropy)
    return mean_entropy


def l2_reg():
    weights_only = filter(lambda x: x.name.endswith('weights:0'), tf.trainable_variables())
    l2_regularization = tf.reduce_sum(tf.stack([tf.nn.l2_loss(x) for x in weights_only]))
    return l2_regularization


def accuracy_of_batch(logits, targets):
    batch_predictions = tf.cast(tf.argmax(logits, 1), tf.int32)
    predicted_correctly = tf.equal(batch_predictions, targets)
    accuracy = tf.reduce_mean(tf.cast(predicted_correctly, tf.float32))
    return accuracy


def get_variables_to_restore():
    checkpoint_exclude_scopes = ["resnet_v1_50/logits"]
    exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)
    return variables_to_restore


def loss(logits, targets):
    logits = tf.squeeze(logits)
    targets = tf.squeeze(tf.cast(targets, tf.int32))

    tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=targets)
    total_loss = tf.losses.get_total_loss()
    total_loss_mean = tf.reduce_mean(total_loss, name='total_loss')

    return total_loss_mean


def accuracy_of_batch2(logits, targets):
    logits = tf.squeeze(logits)
    targets = tf.squeeze(tf.cast(targets, tf.int32))
    batch_predictions = tf.cast(tf.argmax(logits, 1), tf.int32)
    predicted_correctly = tf.equal(batch_predictions, targets)
    accuracy = tf.reduce_mean(tf.cast(predicted_correctly, tf.float32))
    return accuracy

def main( ):

    # Dataset path
    train_tfrecords = 'train.tfrecords'
    test_tfrecords = 'test.tfrecords'
    vgg_ckpt_path = 'checkpoint/resnet_v1_50.ckpt'

    save_ckpt_path = 'checkpoints/my-model.ckpt'
    log_dir = 'log'
    # Learning params
    learning_rate_ini = 0.0001
    training_iters = 50000
    batch_size = 32    # Load batch
    train_img, train_label = input_pipeline(train_tfrecords, batch_size)
    test_img, test_label = input_pipeline(test_tfrecords, batch_size, is_train=False)

    # Model
    #train_instance_pooling, _, _ = vc_net(train_img, is_training=True)
    #test_instance_pooling, _, _ = vc_net(test_img, reuse=True, is_training=False)
    
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        train_logits, end_points1 = resnet_v1.resnet_v1_50(train_img, num_classes=None, is_training=True,global_pool=False)
        test_logits, end_points2 = resnet_v1.resnet_v1_50(test_img, num_classes=None, is_training=False,
                                                         reuse=True, global_pool=False)
    
    fmap1 = end_points1['resnet_v1_50/block4/unit_3/bottleneck_v1']
    fmap2 = end_points2['resnet_v1_50/block4/unit_3/bottleneck_v1']
    
    train_instance_pooling, rest_train, _ = AGOS(fmap1, is_training=True)
    test_instance_pooling, rest_test, _ = AGOS(fmap2, reuse=True, is_training=False)

    
    # Loss and optimizer
    bag_loss = bag_ce(train_instance_pooling, train_label)
    
    rr_loss_train = bag_ce(rest_train, train_label)
    
    l2_regularization = l2_reg()

    ### 0.0005

    ### train loss
    
    a=0.00005

    loss_sum = bag_loss + a*rr_loss_train + 0.00005 * l2_regularization
    
    #loss_sum = bag_loss  + 0.00005 * l2_regularization

    ### test loss
    bag_loss_test = bag_ce(test_instance_pooling, test_label)
    
    rr_loss_test = bag_ce(rest_test , test_label)
    
    l2_regularization = l2_reg()

    loss_sum_test = bag_loss_test + a*rr_loss_test + 0.00005 * l2_regularization
    
    #loss_sum_test = bag_loss_test + 0.00005 * l2_regularization

    global_step = tf.Variable(0, trainable=False)
    global_step_add = global_step.assign_add(1)
    learning_rate = tf.train.exponential_decay(learning_rate_ini, global_step, 5000, 0.5, staircase=True)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_sum)

    # Evaluation
    train_accuracy = accuracy_of_batch(train_instance_pooling, train_label)
    test_accuracy = accuracy_of_batch(test_instance_pooling, test_label)
    
    # Init
    init = tf.global_variables_initializer()

    # Summary
    tf.summary.scalar('bag_ce_loss', bag_loss)
    tf.summary.scalar("train accuracy", train_accuracy)
    tf.summary.scalar("test accuracy", test_accuracy)

    merged_summary_op = tf.summary.merge_all()

    # Create Saver
    variables_to_restore = get_variables_to_restore()
    restorer = tf.train.Saver(variables_to_restore)
    saver = tf.train.Saver(max_to_keep = 2000)

    # Launch the graph
    with tf.Session() as sess:
        print('Init variable')
        sess.run(init)

        restorer.restore(sess, vgg_ckpt_path)
        print('load from ResNet pretrained model')

        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

        file1 = open("lossrecord.txt", 'w')

        print('Start training')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for step in range(training_iters):
            step += 1
            #_, _, bag_ce_value = sess.run([global_step_add, train_op, bag_loss])
            #print('Generation {}: Bag Loss = {:.5f}'.format(step, bag_ce_value))

            ### 2020-01-31 test loss
            _, _, bag_ce_value, bag_ce_value_test = sess.run([global_step_add, train_op, bag_loss, bag_loss_test])
            print('Generation {}: Train Loss = {:.5f}   Test Loss={:.5f}'.format(step, bag_ce_value, bag_ce_value_test))

            with open('lossrecord.txt', 'a') as file1:
                file1.write(str(step) + ' ')
                file1.write("{:.2f}".format(bag_ce_value) + ' ')
                file1.write("{:.2f}".format(bag_ce_value_test) + '\n')

            # Display status
            if step % 40 == 0:
                acc1, acc2, summary_str = sess.run([train_accuracy, test_accuracy, merged_summary_op])
                print(' --- Train Accuracy = {:.2f}%.'.format(100. * acc1))
                print(' --- Test Accuracy = {:.2f}%.'.format(100. * acc2))
                summary_writer.add_summary(summary_str, global_step=step)
            if step % 1000 == 0:
                saver.save(sess, save_ckpt_path, global_step=step)

        coord.request_stop()
        coord.join(threads)
        file1.close()
        print("Finish!")
    tf.reset_default_graph()


if __name__ == '__main__':
    main( )
