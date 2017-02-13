import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def init_weight(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def init_bias(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

batch_size = 100
n_lstm_steps = 1
image_shape = [28,28]
context_shape = [32,32]
dim_context = 32
dim_embed = 16
dim_hidden = 32
W_conv1 = init_weight([5, 5, 1, 32])
b_conv1 = init_bias([32])
W_conv2 = init_weight([5, 5, 32, 64])
b_conv2 = init_bias([64])
W_fc1 = init_weight([7 * 7 * 64, 1024])
b_fc1 = init_bias([1024])
init_hidden_W = init_weight([dim_context, dim_hidden])
init_hidden_b = init_bias([dim_hidden])
init_memory_W = init_weight([dim_context, dim_hidden])
init_memory_b = init_bias([dim_hidden])
context_att_W = init_weight([dim_context, dim_context])
hidden_att_W = init_weight([dim_hidden, dim_context])
pre_att_b = init_bias([dim_context])
lstm_W = init_weight([dim_embed, dim_hidden*4])
lstm_U = init_weight([dim_hidden, dim_hidden*4])
lstm_b = init_bias([dim_hidden*4])
att_W = init_weight([dim_context, 1])
att_b = init_bias([1])
context_encode_W = init_weight([dim_context, dim_hidden*4])
decode_lstm_W = init_weight([dim_hidden, dim_embed])
decode_lstm_b = init_bias([dim_embed])
decode_word_W = init_weight([dim_embed, 10])
decode_word_b = init_bias([10])

def get_initial_lstm(mean_context):
  initial_hidden = tf.nn.tanh(tf.matmul(mean_context, init_hidden_W) + init_hidden_b)
  initial_memory = tf.nn.tanh(tf.matmul(mean_context, init_memory_W) + init_memory_b)
  return initial_hidden, initial_memory

def model(image):
	image = tf.reshape(image, [-1, 28, 28, 1])

	# CNN
	h_conv1 = tf.nn.conv2d(image, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
	h_conv1 = tf.nn.relu(h_conv1 + b_conv1)
	h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	h_conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
	h_conv2 = tf.nn.relu(h_conv2 + b_conv2)
	h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
	context = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

	# attention based RNN
	context = tf.reshape(context, [-1, context_shape[0], context_shape[1]])
	context_flat = tf.reshape(context, [-1, dim_context])
	context_encode = tf.matmul(context_flat, context_att_W)
	context_encode = tf.reshape(context_encode, [-1, context_shape[0], context_shape[1]])
	
	h = tf.nn.tanh(tf.matmul(tf.reduce_mean(context, 1), init_hidden_W) + init_hidden_b)
	c = tf.nn.tanh(tf.matmul(tf.reduce_mean(context, 1), init_memory_W) + init_memory_b)

	for i in range(n_lstm_steps):
		context_encode = context_encode + tf.expand_dims(tf.matmul(h, hidden_att_W), 1) + pre_att_b
		context_encode = tf.nn.tanh(context_encode)
		context_encode_flat = tf.reshape(context_encode, [-1, dim_context])

		alpha = tf.matmul(context_encode_flat, att_W) + att_b
		alpha = tf.reshape(alpha, [-1, context_shape[0]])
		alpha = tf.nn.softmax(alpha)
		weighted_context = tf.reduce_sum(context * tf.expand_dims(alpha, 2), 1)

		emb = tf.zeros([batch_size, dim_embed])
		x_t = tf.matmul(emb, lstm_W) + lstm_b
		lstm_preactive = tf.matmul(h, lstm_U) + tf.matmul(weighted_context, context_encode_W) + x_t

		i, f, o, g = tf.split(1, 4, lstm_preactive)
		i = tf.nn.sigmoid(i)
		f = tf.nn.sigmoid(f)
		o = tf.nn.sigmoid(o)
		g = tf.nn.tanh(g)

		c = f * c + i * g
		h = o * tf.nn.tanh(c)

		logits = tf.matmul(h, decode_lstm_W) + decode_lstm_b
		logits = tf.nn.relu(logits)
		logits = tf.nn.dropout(logits, 0.5)
		label_predict = tf.nn.softmax(tf.matmul(logits, decode_word_W) + decode_word_b)

	return label_predict

# model training
mnist = input_data.read_data_sets("data", one_hot=True)
sess = tf.InteractiveSession()

image = tf.placeholder(tf.float32, [None, 784])
label = tf.placeholder(tf.float32, [batch_size, 10])
label_predict = model(image)

cross_entropy = -tf.reduce_sum(label * tf.log(label_predict))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.arg_max(label_predict, 1), tf.arg_max(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables())

for i in range(1000):
  batch = mnist.train.next_batch(batch_size)
  #context_batch = numpy.random.random((100, 28, 28))
  #print(context_batch.shape)
  #exit()
  if i % 100 == 0:
    train_accuacy = accuracy.eval(feed_dict={image: batch[0], label: batch[1]})
    print("step %d, training accuracy %g"%(i, train_accuacy))
  train_step.run(feed_dict = {image: batch[0], label: batch[1]})

# accuracy on test
test_accuracy = 0.0
for i in range(100):
  batch = mnist.test.next_batch(batch_size)
  temp_accuacy = accuracy.eval(feed_dict={image: batch[0], label: batch[1]})
  test_accuracy += temp_accuacy
print("test accuracy: %g"%(test_accuracy/10))
