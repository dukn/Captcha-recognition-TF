import cPickle,numpy as np,Image,gzip,pylab
from scipy import signal
from scipy import misc
#sub import 
import matplotlib.pyplot as plt
from PIL import Image

prepath = 'Captcha/lv1/'

X = []
for Im in range(15000):
	path = prepath+"%d.jpg"%(Im)
	img = misc.imread(path).astype(np.float)
	grayim = np.dot(img[...,:3],[0.299,0.587,0.114])
	X.append(grayim)

	if Im%20 == 0:
		print '.',
	if Im%1000 == 0:
		print
print
print 'Compressing...'
with gzip.open("captcha1.pkl.gz", 'wb') as f:
	cPickle.dump(X,f)

print
print 'completed!'
