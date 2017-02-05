import tensorflow as tf
import numpy as np
import pickle,numpy,gzip
import datetime as dt
import matplotlib.pyplot as plt
import pylab
from scipy import signal
from scipy import misc
from PIL import Image

prepath = 'Captcha/lv3/'

convert = ['a','b','c','d','e','f','g','t','u','v','w','x','y','z','1','2','3','4','5','6']
revert = {'a':0,'b':1,'c':2,'d':3,'e':4,'f':5,'g':6,'t':7,'u':8,'v':9,
	'w':10,'x':11,'y':12,'z':13,'1':14,'2':15,'3':16,'4':17,'5':18,'6':19}

def Num2Char(n): 
	return convert[n];

def Char2Num(c):
	return revert[c]

def y_out(seq):
	arr = []
	for i in seq:
		arr.append(Char2Num(i))
	arr2 = [0.0 for i in range(20*5)]
	for i in range(len(arr)):
		arr2[i*20+arr[i]] = 0.2
	return arr2

dataX = []
for Im in range(100):
	path = prepath+"%d.jpg"%(Im)
	img = misc.imread(path).astype(np.float)
	grayim = np.dot(img[...,:3],[0.299,0.587,0.114])
	dataX.append(grayim)

f2 = open('Captcha/pass3.txt')
labelY = f2.read().split('\n')[:100]

dataY = []

for i in range(len(labelY)):
	dataY.append(y_out(labelY[i]))

# delete 
del labelY
reX = dataX
reY = np.array(dataY)

#exit(0)
learning_rate = 0.001
batch_size = 50
training_iters = 2500  # 128*5000
display_step = 10
dropout = 0.75
# define placeholder
X = tf.placeholder(tf.float32,[None,40,150,1])
Y = tf.placeholder(tf.float32,[None,100])

def weight_variable(shape):
	initial = tf.truncated_normal(shape,stddev = 0.2)
	return tf.Variable(initial)
def bias_variable(shape):
	initial = tf.constant(0.5,shape=shape)
	return tf.Variable(initial)

#probability placeholder for dropout layer
keep_prob = tf.placeholder(tf.float32,name='keep_prob')

sess = tf.Session()
def conv2d(img, w, b):
	return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img,w,strides=[1,1,1,1],padding='SAME'),b))
def max_pool(img, k):
	return tf.nn.max_pool(img,ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME')

def make_model(_X, _weights, _biases, _dropout):
	# First we have Convolution layer 5x5x48 with relu
	conv1 = conv2d(_X,_weights['wc1'],_biases['bc1'])

	# Max Pooling (down-sampling),change input size by factor of 2
	conv1 = max_pool(conv1,k=2)

	# Apply Dropout
	conv1 = tf.nn.dropout(conv1,_dropout)

	# Then Convolution layer 5x5x64
	conv2 = conv2d(conv1,_weights['wc2'],_biases['bc2'])

	# Max Pooling 
	conv2 = max_pool(conv2,k=2)

	# Apply Dropout
	conv2 = tf.nn.dropout(conv2,_dropout)

	# Convolution layer 5x5x128
	conv3 = conv2d(conv2,_weights['wc3'],_biases['bc3'])

	# Max Pooling
	conv3 = max_pool(conv3,k=2)

	#Apply dropout
	conv3 = tf.nn.dropout(conv3,_dropout)

	#Reshape and do fully-connected hidden layer using matrix multiplication
	pool_shape= conv3.get_shape().as_list()
	fullcn = tf.reshape(conv3,[-1,pool_shape[1]* pool_shape[2]*pool_shape[3]] )

	fullcn = tf.nn.relu(
		tf.add(tf.matmul(fullcn,_weights['wd1']), _biases['bd1'])
		)
	fullcn = tf.nn.dropout(fullcn, _dropout)

	# Output
	out = tf.add(tf.matmul(fullcn,_weights['out']),_biases['out'])

	return out  

weight = {
	'wc1': tf.Variable(tf.truncated_normal([5,5,1,32],stddev = 0.1)),
	'wc2':tf.Variable(tf.truncated_normal([5,5,32,48], stddev = 0.1)),
	'wc3':tf.Variable(tf.truncated_normal([5,5,48,64], stddev = 0.1)),
	'wd1':tf.Variable(tf.truncated_normal([5*19*64,1000], stddev = 0.1)),
	'out':tf.Variable(tf.truncated_normal([1000,100], stddev = 0.1))
}

biases = {
	'bc1': tf.Variable(0.1*tf.random_normal([32])),
	'bc2': tf.Variable(0.1*tf.random_normal([48])),
	'bc3': tf.Variable(0.1*tf.random_normal([64])),
	'bd1': tf.Variable(0.1*tf.random_normal([1000])),
	'out': tf.Variable(0.1*tf.random_normal([100]))
}

# Construct model
predict = make_model(X,weight,biases,keep_prob)

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(predict,Y)
loss = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# batch, rows, cols
p = tf.reshape(predict,[-1,5,20])
# max idx acros the rows
max_idx_p = tf.argmax(p,2)

l = tf.reshape(Y,[-1,5,20])
max_idx_l = tf.argmax(l,2)

correct_pred = tf.equal(max_idx_p,max_idx_l)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

losses = list()
accuracies = list()
saver = tf.train.Saver()

def showcapt(arr):
	res = ''
	achar = -1
	#print 'debug in show captcha'
	#print arr
	#print arr[0]
	for i in range(5):
		achar = arr[i]
		res+=Num2Char(achar)
	return res 

from scipy import signal
from scipy import misc
#sub import 
import matplotlib.pyplot as plt
from PIL import Image

#print '  training...'
# Launch the graph
def view_(_pred,_lable):
	
	fname = ['Captcha/lv3/%i.jpg' %i for i in range(20)]
	img = []
	for fn in fname:
		img.append(Image.open(open(fn)))
		#img.append(misc.imread(fn).astype(np.float))
	for i in range(len(img)):
		pylab.subplot(4,5,i+1); pylab.axis('off')
		
		pylab.imshow(img[i])
		#pylab.imshow( np.dot(np.array(img[i])[...,:3],[0.299,0.587,0.114]) , cmap=plt.get_cmap("gray"))
		#pylab.text(40,60,_pred[i],color = 'b')
		if ( _pred[i] == _lable[i] ):
			pylab.text(40,65,_pred[i],color = 'b',size = 15)
		else:
			pylab.text(40,65,_pred[i],color = 'r',size = 15)
		
		pylab.text(40,92,_lable[i],color = 'g',size = 15)

	pylab.show()		

with tf.Session() as sess:
	#load old model
	#saver.restore(sess, "model.data")
	new_saver = tf.train.import_meta_graph('train3.data.meta')
	new_saver.restore(sess, tf.train.latest_checkpoint('./'))
	all_vars = tf.trainable_variables()

	#sess.run(init)
	step = 0
	epoch = 0
	start_epoch = dt.datetime.now()
	#training
	for step in range(1):
		#Optimizer model by training
		#for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
		#	sess.run(optimizer,feed_dict={X:np.array(trX[start:end]).reshape(-1,40,140,1)/256.0,Y:trY[start:end],keep_prob:dropout})
		
		# Calculate accuracy
		acc = 0
		batch_loss = 0
		for start, end in zip(range(0, len(reX), batch_size), range(batch_size, len(reX), batch_size)):
			acc = sess.run(accuracy,feed_dict={X:np.array(reX[start:end]).reshape(-1,40,150,1)/256.0,Y:reY[start:end],keep_prob:1.})
			accuracies.append(acc)
			#
			batch_loss=sess.run(loss,feed_dict={X:np.array(reX[start:end]).reshape(-1,40,150,1)/256.0,Y:reY[start:end],keep_prob:1.})
			losses.append(batch_loss)
		#print '#Step %d, loss = %.4f, accuracy = %.4f'%(step,batch_loss,acc)
		#print '\t\t@losses = %.4f, accuracies = %.4f'%(sum(losses)/len(losses), sum(accuracies)/len(accuracies)	)
		
		# Show word predict
		pp = sess.run(predict,feed_dict={X:np.array(reX[:batch_size]).reshape(-1,40,150,1)/256.0,Y:reY[:batch_size],keep_prob:1.})
		p = tf.reshape(pp, [batch_size,5,20])
		max_idx_p = tf.argmax(p,2).eval()
		
		ff = reY[:batch_size]
		f = tf.reshape(ff,[batch_size,5,20])
		max_idx_f = tf.argmax(f,2).eval()

		v_p = []
		v_f = []		
		#print 'prediction and fact : '
		for tmp in range(20):
			v_p.append(showcapt(max_idx_p[tmp]))
		#print
		for tmp in range(20):	
			v_f.append(showcapt(max_idx_f[tmp]))
		#print

		view_(v_p,v_f)
		#if step % 10 == 0:
		#	saver.save(sess, "hard_train.data")

