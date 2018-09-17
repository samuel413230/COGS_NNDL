# Samuel Sunarjo A12340230

# Problem 1

import numpy as np
A = np.array([[1,3],[5,7],[9,11]])
B = np.array([[1,-1],[-1,1],[-1,0]])

A-B
A*B
np.dot(np.transpose(A),B)
np.dot(A,np.transpose(B))
np.dot(A,B)


# Problem 2

import matplotlib.pyplot as plt
%matplotlib inline
np.random.seed(0)
space = np.linspace(0,10,num=50)
sine = np.sin(space)
plt.scatter(space,sine,color='b',label = 'sine_curve')
sine_5 = sine
for i in range(5):
    sine_5 = sine_5 + np.random.normal(scale=0.1,size=50)
plt_sine_5, = plt.plot(space,sine_5,color='r',label = 'noise_5_iters')

# 20 iteration
sine_20 = sine
for i in range(20):
    sine_20 = sine_20 + np.random.normal(scale=0.1,size=50)
plt_sine_20, = plt.plot(space,sine_20,color='g',label = 'noise_20_iters')

# 100 iteration
sine_100 = sine
for i in range(100):
    sine_100 = sine_100 + np.random.normal(scale=0.1,size=50)
plt_sine_100, = plt.plot(space,sine_100,color='y',label = 'noise_100_iters')

plt.legend(loc='upper right')
plt.savefig('./Q2.png')


# Problem 3

img.shape

import scipy
img_300x300 = scipy.misc.imresize(img,(300,300))
img_300x300.shape
#scipy.misc.imshow(img_300x300)
scipy.misc.imsave("tabby300x300.png",img_300x300)

from scipy.ndimage import rotate
rotated1 = rotate(img,30,reshape=True,mode='nearest',cval=0)
scipy.misc.imsave("tabby_rotated1.png",rotated1)

rotated2 = rotate(img,30,reshape=False,mode='constant',cval=0)
scipy.misc.imsave("tabby_rotated2.png",rotated2)

rotated3 = rotate(img,30,reshape=False,mode='constant',cval=255)
scipy.misc.imsave("tabby_rotated3.png",rotated3)

rotated4 = rotate(img,30,reshape=False,mode='reflect',cval=0)
scipy.misc.imsave("tabby_rotated4.png",rotated4)


# Problem 4

import matplotlib.pyplot as plt
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D

iris = datasets.load_iris()
X = iris.data
Y = iris.target

# 2 feature
plt.scatter(X[:,0],X[:,1], c=Y, cmap=plt.cm.Paired)
plt.savefig('./Q4_2feature.png')

# 3 feature
fig = plt.figure(1, figsize=(8, 8))
ax = Axes3D(fig, elev=-150, azim=110)
ax.scatter(X[:,0],X[:,1],X[:,2],c=Y,cmap=plt.cm.Paired)


# Problem 5

np.array([[182.3, 62, 1, 0, 0, 0],
       [181, 66, 0, 1, 0, 0],
       [186, 56, 0, 0, 1, 0],
       [179, 59, 0, 1, 0, 0],
       [182, 50, 0, 0, 0, 1]
      ])


# Problem 6

import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt('data.txt',dtype='float')
x = data[:,0].reshape(len(data),1)
y = data[:,1].reshape(len(data),1)

plt.plot(x,y)
plt.grid()

X = np.hstack((np.ones((len(x),1)),np.power(x,1)))

X_t = X.transpose((1,0))
sol = np.dot(np.linalg.inv(np.dot(X_t,X)),np.dot(X_t,y))

plt.plot(x,y)
#plt.hold(True)
plt.plot(x,sol[0]+sol[1]*x)
plt.title('Least square line fitting')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('./Q6.png')
