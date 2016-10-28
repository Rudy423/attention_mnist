import math
import numpy
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def init_weight(dim_in, dim_out, name=None, stddev=1.0):
  return tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=stddev/math.sqrt(float(dim_in))), name=name)

def init_bias(dim_out, name=None):
  return tf.Variable(tf.zeros([dim_out]), name=name)

batch_size = 100
n_lstm_steps = 1
ctx_shape = [28,28]
dim_ctx = 28
dim_embed = 14
dim_hidden = 28
init_hidden_W = init_weight(dim_ctx, dim_hidden)
init_hidden_b = init_bias(dim_hidden)
init_memory_W = init_weight(dim_ctx, dim_hidden)
init_memory_b = init_bias(dim_hidden)
image_att_W = init_weight(dim_ctx, dim_ctx)
hidden_att_W = init_weight(dim_hidden, dim_ctx)
pre_att_b = init_bias(dim_ctx)
lstm_W = init_weight(dim_embed, dim_hidden*4)
lstm_U = init_weight(dim_hidden, dim_hidden*4)
lstm_b = init_bias(dim_hidden*4)
att_W = init_weight(dim_ctx, 1)
att_b = init_bias(1)
image_encode_W = init_weight(dim_ctx, dim_hidden*4)
decode_lstm_W = init_weight(dim_hidden, dim_embed)
decode_lstm_b = init_bias(dim_embed)
decode_word_W = init_weight(dim_embed, 10)
decode_word_b = init_bias(10)

def get_initial_lstm(mean_image):
  initial_hidden = tf.nn.tanh(tf.matmul(mean_image, init_hidden_W) + init_hidden_b)
  initial_memory = tf.nn.tanh(tf.matmul(mean_image, init_memory_W) + init_memory_b)
  return initial_hidden, initial_memory

def predict(image):
	h, c = get_initial_lstm(tf.reduce_mean(image, 1))
	image_flat = tf.reshape(image, [-1, dim_ctx])
	image_encode = tf.matmul(image_flat, image_att_W)
	image_encode = tf.reshape(image_encode, [-1, ctx_shape[0], ctx_shape[1]])

	image_encode = image_encode + tf.expand_dims(tf.matmul(h, hidden_att_W), 1) + pre_att_b
	image_encode = tf.nn.tanh(image_encode)
	image_encode_flat = tf.reshape(image_encode, [-1, dim_ctx])

	alpha = tf.matmul(image_encode_flat, att_W) + att_b
	alpha = tf.reshape(alpha, [-1, ctx_shape[0]])
	alpha = tf.nn.softmax(alpha)
	weighted_image = tf.reduce_sum(image * tf.expand_dims(alpha, 2), 1)

	lstm_preactive = tf.matmul(h, lstm_U) + tf.matmul(weighted_image, image_encode_W)
	i, f, o, new_c = tf.split(1, 4, lstm_preactive)

	i = tf.nn.sigmoid(i)
	f = tf.nn.sigmoid(f)
	o = tf.nn.sigmoid(o)
	new_c = tf.nn.tanh(new_c)

	c = f * c + i * new_c
	h = o * tf.nn.tanh(new_c)

	logits = tf.matmul(h, decode_lstm_W) + decode_lstm_b
	logits = tf.nn.relu(logits)

	label_predict = tf.nn.softmax(tf.matmul(logits, decode_word_W) + decode_word_b)
	return label_predict


# model training
mnist = input_data.read_data_sets("data", one_hot=True)
sess = tf.InteractiveSession()

image = tf.placeholder(tf.float32, [batch_size, ctx_shape[0], ctx_shape[1]])
label = tf.placeholder(tf.float32, [batch_size, 10])
label_predict = predict(image)

cross_entropy = -tf.reduce_sum(label * tf.log(label_predict))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.arg_max(label_predict, 1), tf.arg_max(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables())

for i in range(1000):
    batch = mnist.train.next_batch(batch_size)
    image_batch = batch[0].reshape([-1, ctx_shape[0], ctx_shape[1]])
    label_batch = batch[1]
    #image_batch = numpy.random.random((100, 28, 28))
    #print(image_batch.shape)
    #exit()
    if i % 100 == 0:
        train_accuacy = accuracy.eval(feed_dict={image: image_batch, label: label_batch})
        print("step %d, training accuracy %g"%(i, train_accuacy))
    train_step.run(feed_dict = {image: image_batch, label: label_batch})

# accuracy on test
myAccuracy = 0.0
for i in range(10):
    batch = mnist.test.next_batch(1000)
    test_accuacy = accuracy.eval(feed_dict={image: image_batch, label: label_batch})
    myAccuracy += test_accuacy
print("test accuracy: %g"%(myAccuracy/10))
