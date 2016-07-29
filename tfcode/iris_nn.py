__author__ = 'Pascalito'

import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression


class IrisNN(object):

    def __init__(self):
        # DEFINING APPLICATION
        # defining input to application
        self.x_tensor = tf.placeholder(dtype=tf.float32, shape=(None, 4))
        self.y_tensor = tf.placeholder(dtype=tf.int32, shape=None)
        # defining logistic regression pipeline
        W1 = tf.Variable(dtype=tf.float32, name='W', initial_value=tf.random_normal(mean=0, stddev=0.01, shape=[4, 3]))
        b1 = tf.Variable(dtype=tf.float32, name='b', initial_value=tf.random_normal(mean=0, stddev=0.01, shape=[3]))
        self.log_reg_tensor = tf.matmul(self.x_tensor, W1) + b1

        # DEFINING OPTIMIZATION PROBLEM
        # defining loss
        individual_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(self.log_reg_tensor, self.y_tensor)
        regularizer = tf.nn.l2_loss(W1)
        self.loss = tf.reduce_mean(individual_losses, 0) + 0.001 * regularizer
        # attaching solver
        optimizer = tf.train.AdamOptimizer()
        self.opt_pbm_sol = optimizer.minimize(self.loss)

        # DEFINING PREDICTED TENSOR
        self.predicted_class = tf.arg_max(self.log_reg_tensor, 1)


def run_tensorflow_log_reg():
    iris_data = load_iris()
    x, y = iris_data.data, iris_data.target
    nn = IrisNN()

    with tf.Session() as session:
        # inializing variables of that session
        init = tf.initialize_all_variables()
        session.run(init)
        # performing SGD
        for i in range(1, 10000):
            shuf_x, shuf_y = shuffle(x, y)
            feed_dict = {nn.x_tensor: shuf_x, nn.y_tensor: shuf_y}
            opt_sol, loss = session.run([nn.opt_pbm_sol, nn.loss], feed_dict)
            print loss
        # pulling predictions
        feed_dict = {nn.x_tensor: x, nn.y_tensor: y}
        predictions, layer1 = session.run([nn.predicted_class,  nn.log_reg_tensor], feed_dict)
        # printing results
        conf_mat = confusion_matrix(y, predictions)
        print conf_mat
        summarry_writer = tf.train.SummaryWriter(logdir="/tmp/tensor_flow_intro/iris/", graph=session.graph)
        summarry_writer.flush()



def run_scikit_log_reg():
    iris_data = load_iris()
    x, y = iris_data.data, iris_data.target
    logistic = LogisticRegression()
    logistic.fit(x, y)
    predictions = logistic.predict(x)
    conf_mat = confusion_matrix(y, predictions)
    print conf_mat

if __name__ == "__main__":
    print "Hello deep-learning world!"
    run_tensorflow_log_reg()
    #run_scikit_log_reg()

