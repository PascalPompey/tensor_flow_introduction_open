__author__ = 'Pascalito'


import tensorflow as tf
import os

def hello_data():
    nb_values = 10

    # Defining inputs to application
    input_placeholder = tf.placeholder(name='InputPlaceholder', dtype=tf.int32, shape=[nb_values])
    # Defining application behavior
    values_plus_1_tensor = tf.add(input_placeholder, 1)

    # registering input data
    values = list(range(0, nb_values))
    feed_dict = {input_placeholder: values}

    # Running application (in session)
    with tf.Session() as session:
        # running and fetching values of interest
        application_output = session.run(values_plus_1_tensor, feed_dict)
        print application_output
        summarry_writer = tf.train.SummaryWriter(logdir="/tmp/tensor_flow_intro/hello_data/", graph=session.graph)
        summarry_writer.flush()


if __name__ == "__main__":
    print('Hello deep-learning world!')
    hello_data()
    # source activate stan
    #tensorboard --logdir /tmp/tensor_flow_intro

