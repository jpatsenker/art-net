import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import scipy.ndimage.interpolation
from keras.datasets import mnist
import keras
import pandas as pd
from scipy.misc import imread
import urllib
import time

#use "nvidia-smi" to see what devices are in use
os.environ["CUDA_VISIBLE_DEVICES"]="2"

if not os.path.exists('out/'):
    os.makedirs('out/')

''' Helper functions '''


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)
    
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
    
    return fig

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def log(x):
    return tf.log(x + 1e-8)

def sample_X(X, size):
    start_idx = np.random.randint(0, X.shape[0]-size)
    return X[start_idx:start_idx+size]

def sample_XY(X,y,size):
	start_idx = np.random.randint(0, X.shape[0]-size)
	return X[start_idx:start_idx+size], y[start_idx:start_idx+size]

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)

def bn_lrelu(x, phase, scope):
    with tf.variable_scope(scope):
        x_bn = tf.contrib.layers.batch_norm(x, center=True, scale=True, is_training=phase, scope='bn')
        return lrelu(x_bn, name = 'lrelu')
    
def bn_sigmoid(x, phase, scope):
    with tf.variable_scope(scope):
        x_bn = tf.contrib.layers.batch_norm(x, center=True, scale=True, is_training=phase, scope='bn')
        return tf.nn.sigmoid(x_bn, 'sigmoid')

def bn_softplus(x,phase,scope):
	with tf.variable_scope(scope):
		x_bn = tf.contrib.layers.batch_norm(x, center=True, scale=True, is_training=phase, scope='bn')
		return tf.nn.softplus(x_bn, name='softplus')

def augment(X,y,num):
	#######AUGMENT THE DATA
	datagen = ImageDataGenerator(
	    rotation_range=360.,
	    horizontal_flip=True,
	    vertical_flip=True,
	    data_format="channels_last")

	flow = datagen.flow(X[:,:,:,:], y[:], batch_size=augment_batch)

	synth_train_images = X
	synth_labels = y

	count = 0

	for (xa, ya) in flow:
		#map(lambda a: draw_image.drawImageFromTensor(x[a,:,:,:],0,gray=(0,1)), range(x.shape[0]))
		synth_train_images = np.concatenate([synth_train_images, xa], 0)
		synth_labels = np.concatenate([synth_labels, ya], 0)
		count+=1
		if count==augment_num:
			break
	return synth_train_images,synth_labels

def getRandomPatches(tensor, x, y, num):
	patches = np.zeros([num,x,y,tensor.shape[3]])
	for n in range(num):
		p = (np.random.rand(1)*tensor.shape[0]).astype('int32')
		coords = np.random.rand(2)
		xcoord = ((tensor.shape[1]-x)*coords[0]).astype('int32')
		ycoord = ((tensor.shape[2]-y)*coords[1]).astype('int32')
		patches[n] = np.array(tensor[p,xcoord:xcoord+x,ycoord:ycoord+y])
	return patches


def read_meta(file):
    data = pd.read_csv("data/wga/catalog.csv",delimiter=";", quotechar='"')
    return np.array(data)

def getImage(link):
    print link
    l = urllib.urlopen(link)
    img=imread(l)
    del l
    return img


#http://www.wga.hu/html/a/aachen/allegory.html -> http://www.wga.hu/art/a/aachen/allegory.jpg
def extractImageLink(link):
    link_arr = link.split("/")
    link_arr[3] = 'art'
    link_arr[-1] = link_arr[-1].replace('.html','.jpg')
    new_link = '/'.join(link_arr)
    return new_link

SD = {
        'Austrian'      :   0,
        'American'      :   1,
        'Belgian'       :   2,
        'British'       :   3,
        'Bohemian'      :   4,
        'Catalan'       :   5,
        'Dutch'         :   6,
        'Danish'        :   7,
        'English'       :   8,
        'Finnish'       :   9,
        'Flemish'       :   10,
        'French'        :   11,
        'German'        :   12,
        'Greek'         :   13,
        'Hungarian'     :   14,
        'Italian'       :   15,
        'Irish'         :   16,
        'Netherlandish' :   17,
        'Norwegian'     :   18,
        'Polish'        :   19,
        'Portuguese'    :   20,
        'Russian'       :   21,
        'Scottish'      :   22,
        'Spanish'       :   23,
        'Swedish'       :   24,
        'Swiss'         :   25,
        'Other'         :   26
    }
def style_dictionary(style):
    return SD[style]




'''
Take in data as a 4D tensor of shape [num_slides/num_slide_patches, slide_xdim, slide_ydim, channels]
'''
#PARAMS
DATA_FILE = "data/wga/catalog.csv"

# load data
# (X_train, Y_train), (_, _) = mnist.load_data()  
# X_train = X_train.reshape(X_train.shape[0],  28, 28, 1)/255. 
# Y_train_logits = keras.utils.to_categorical(Y_train)
# Y_train = Y_train.astype("int32")


meta_data = read_meta(DATA_FILE)

rel_meta_data = meta_data[np.where(meta_data[:,7]=='painting')]

image_links = rel_meta_data[:,6]

image_styles = rel_meta_data[:,9] #school

#image_subjects = rel_meta_data[:,8] #subject

image_labels = np.array(map(style_dictionary,image_styles))

Y_train = image_labels

Y_train_logits = keras.utils.to_categorical(Y_train).astype('int32')

#data = X_train
#init_func=tf.truncated_normal

extracted_image_links = np.array(map(lambda link: extractImageLink(link),image_links))

store = np.zeros([image_links.shape[0],2])
t = time.time()
for i in range(extracted_image_links.shape[0]):
    img = getImage(extracted_image_links[i])
    store[i,:] = np.array(img.shape[0:2])
    del img
    if i%1==0:
        print i, time.time()-t

np.savetxt("test.txt",store)
exit(1)



n_rows = data.shape[1]
n_cols = data.shape[2]
numChannels = data.shape[3]
numClasses = Y_train_logits.shape[1]
batch_size=16

#Setup Input Layers
input_layer = tf.placeholder(tf.float32, shape=[None, n_rows, n_cols, numChannels])
tlabels = tf.placeholder(tf.int32, shape=[None])
phase = tf.placeholder(tf.bool, name='phase')

'''Initialize All Weights'''


#####ENCODER
#First Convolution Layer
conv1_weights = tf.Variable(init_func([5, 5, numChannels, 32], stddev=0.1), name="conv1_weights")
conv1_biases = tf.Variable(tf.zeros([32]), name="conv1_biases")

#Second Convolution Layer
conv2_weights = tf.Variable(init_func([5, 5, 32, 64], stddev=0.1), name="conv2_weights")
conv2_biases = tf.Variable(tf.zeros([64]), name="conv2_biases")

#Third Convolution Layer
conv3_weights = tf.Variable(init_func([5, 5, 64, 128], stddev=0.1), name="conv3_weights")
conv3_biases = tf.Variable(tf.zeros([128]), name="conv3_biases")

#Fourth Convolution Layer
conv4_weights = tf.Variable(init_func([5, 5, 128, 128], stddev=0.1), name="conv4_weights")
conv4_biases = tf.Variable(tf.zeros([128]), name="conv4_biases")


theta_PRE = [conv1_weights, conv1_biases, 
           conv2_weights, conv2_biases,
           conv3_weights, conv3_biases,
           conv4_weights, conv4_biases,
           dense1_weights, dense1_biases]



shapes_E = []
def FCN_E(X, Yt):
	#track shape
    shapes_E.append(X.get_shape().as_list())

    #run through layer
    h1 = tf.nn.conv2d(X, conv1_weights, strides=[1,2,2,1], padding='SAME', name = 'h1_conv') + conv1_biases

    #run through non-linear transform
    h1 = bn_lrelu(h1, phase, 'E_layer1')

    shapes_E.append(h1.get_shape().as_list())
    h2 = tf.nn.conv2d(h1, conv2_weights, strides=[1,1,1,1], padding='SAME', name = 'h2_conv') + conv2_biases
    h2 = bn_lrelu(h2, phase, 'E_layer2')

    shapes_E.append(h2.get_shape().as_list())
    h3 = tf.nn.conv2d(h2, conv3_weights, strides=[1,1,1,1], padding='SAME', name = 'h3_conv') + conv3_biases
    h3 = bn_lrelu(h3, phase, 'E_layer3')
    shapes_E.append(h3.get_shape().as_list())

    h4 = tf.nn.conv2d(h3, conv4_weights, strides=[1,1,1,1], padding='SAME', name = 'h3_conv') + conv4_biases
    h4 = bn_lrelu(h4, phase, 'FCN_E_output')
    shapes_E.append(h4.get_shape().as_list())

    return h4, Yt


shapes_P = []
def Classifier(X,Y): #take image, labels
	shapes_P.append(X.get_shape().as_list())

	h1 = tf.contrib.layers.flatten(X)

	shapes_P.append(h1.get_shape().as_list())

	h2 = tf.matmul(h1, dense1_weights, name='h2_dense') + dense1_biases

	h2 = bn_softplus(h2, phase, "PRETRAINER_OUTPUT")

	shapes_P.append(h2.get_shape().as_list())

	return h2,Y



def Diabolo_E(X):
	pass

def Diabolo_D(X):
	pass

#print X_train.shape


'''
Data Flow
'''

Z,tY = FCN_E(input_layer, tlabels)
Y,tY = Classifier(Z,tY)


'''
Set Loss
'''
PRE_TRAIN_LOSS = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=Y, labels=tY))

'''
Set Optimizers
'''
eta=1e-2
solver = tf.train.AdamOptimizer(learning_rate=eta)
#FCN_solver = solver.minimize(FCN_RECON_LOSS, var_list=theta_FCN)
PRE_solver = solver.minimize(PRE_TRAIN_LOSS, var_list=theta_PRE)


'''
Start Session
'''

sess = tf.Session()

#initialize variables
sess.run(tf.global_variables_initializer())

import time
tstart=time.time()

numSteps=10000
i=0

for it in range(numSteps):
    # Sample data from both domains
    xa,ya = sample_XY(X_train,Y_train,size=batch_size)
    _, loss_curr = sess.run([PRE_solver, PRE_TRAIN_LOSS], feed_dict={input_layer: xa, tlabels: ya, phase: 1})
    print('Iter: {}; D_loss: {}'.format(it, loss_curr))
    print("Timer: ", time.time()-tstart)
    tstart=time.time()
    inp,act = sample_XY(X_train, Y_train, size=batch_size)
    s = sess.run(Y, feed_dict={input_layer: inp, phase:0})
    s_labels = np.apply_along_axis(np.argmax, 1, s)
    print s_labels,"\n",act,"\n","Num Accurate:", np.sum(s_labels==act)

# for it in range(numSteps):
#     # Sample data from both domains
#     xa = sample_X(X_train,size=batch_size)
#     _, loss_curr = sess.run([FCN_solver, FCN_RECON_LOSS], feed_dict={input_layer: xa, phase: 1})
#     #
#     if it % 100 == 0:
#         print('Iter: {}; D_loss: {:.4}'.format(it, loss_curr))
#         print "Timer: ", time.time()-tstart
#         tstart=time.time()
#         inp = sample_X(X_train, size=batch_size)
#         s = sess.run(Y, feed_dict={input_layer: inp, phase:0})
#         fig = plot(s[0:16,:,:,0])
#         plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
#         i += 1
#         plt.close(fig)
