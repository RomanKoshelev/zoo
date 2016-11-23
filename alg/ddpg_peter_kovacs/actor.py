import tensorflow as tf

HIDDEN1_UNITS = 400
HIDDEN2_UNITS = 300


# noinspection PyPep8Naming
class ActorNetwork(object):
    def __init__(self, sess, state_size, action_size, TAU, LEARNING_RATE, L2):
        self.sess = sess
        self.TAU = TAU
        self.L2 = L2

        with tf.variable_scope("master"):
            self.state, self.out, self.net = self.create_actor_network(state_size, action_size)

        with tf.variable_scope("target"):
            self.target_state, self.target_update, self.target_net, self.target_out = \
                self.crate_actor_target_network(state_size, self.net)

        # TRAINING
        self.action_gradient = tf.placeholder(tf.float32, [None, action_size], name='action_gradient')
        self.params_grad = tf.gradients(
            self.out, self.net, -self.action_gradient)
        grads = zip(self.params_grad, self.net)
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

    def predict(self, states):
        return self.sess.run(self.out, feed_dict={
            self.state: states
        })

    def target_predict(self, states):
        return self.sess.run(self.target_out, feed_dict={
            self.target_state: states
        })

    def target_train(self):
        self.sess.run(self.target_update)

    def crate_actor_target_network(self, input_dim, net):
        state = tf.placeholder(tf.float32, shape=[None, input_dim], name='state')

        ema = tf.train.ExponentialMovingAverage(decay=1 - self.TAU)
        target_update = ema.apply(net)
        target_net = [ema.average(x) for x in net]

        h1 = tf.nn.relu(tf.matmul(state, target_net[0]) + target_net[1])
        h2 = tf.nn.relu(tf.matmul(h1, target_net[2]) + target_net[3])
        out = tf.tanh(tf.matmul(h2, target_net[4]) + target_net[5])

        return state, target_update, target_net, out

    def create_actor_network(self, input_dim, output_dim):
        state = tf.placeholder(tf.float32, shape=[None, input_dim], name='state')

        W1 = self.weight_variable([input_dim, HIDDEN1_UNITS])
        b1 = self.bias_variable([HIDDEN1_UNITS])
        W2 = self.weight_variable([HIDDEN1_UNITS, HIDDEN2_UNITS])
        b2 = self.bias_variable([HIDDEN2_UNITS])
        W3 = self.weight_variable([HIDDEN2_UNITS, output_dim])
        b3 = self.bias_variable([output_dim])

        h1 = tf.nn.relu(tf.matmul(state, W1) + b1)
        h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)
        out = tf.tanh(tf.matmul(h2, W3) + b3)

        return state, out, [W1, b1, W2, b2, W3, b3]

    @staticmethod
    def weight_variable(shape):
        initial = tf.random_uniform(shape, minval=-0.05, maxval=0.05)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)
