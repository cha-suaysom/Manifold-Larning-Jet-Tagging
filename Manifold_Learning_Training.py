def rotate_and_reflect(x,y,w):
    rot_x = []
    rot_y = []
    theta = 0
    maxPt = -1
    for ix, iy, iw in zip(x, y, w):
        dR = np.sqrt(ix*ix+iy*iy)
        thisPt = iw
        if dR > 0.1 and thisPt > maxPt:
            maxPt = thisPt
            # rotation in eta-phi plane c.f  https://arxiv.org/abs/1407.5675 and https://arxiv.org/abs/1511.05190:
            # theta = -np.arctan2(iy,ix)-np.radians(90)
            # rotation by lorentz transformation c.f. https://arxiv.org/abs/1704.02124:
            px = iw*np.cos(iy)
            py = iw*np.sin(iy)
            pz = iw*np.sinh(ix)
            theta = np.arctan2(py,pz)+np.radians(90)
            
    c, s = np.cos(theta), np.sin(theta)
    R = np.matrix('{} {}; {} {}'.format(c, -s, s, c))
    for ix, iy, iw in zip(x, y, w):
        # rotation in eta-phi plane:
        #rot = R*np.matrix([[ix],[iy]])
        #rix, riy = rot[0,0], rot[1,0]
        # rotation by lorentz transformation
        px = iw*np.cos(iy)
        py = iw*np.sin(iy)
        pz = iw*np.sinh(ix)
        rot = R*np.matrix([[py],[pz]])
        px1 = px
        py1 = rot[0,0]
        pz1 = rot[1,0]
        iw1 = np.sqrt(px1*px1+py1*py1)
        rix, riy = np.arcsinh(pz1/iw1), np.arcsin(py1/iw1)
        rot_x.append(rix)
        rot_y.append(riy)
        
    # now reflect if leftSum > rightSum
    leftSum = 0
    rightSum = 0
    for ix, iy, iw in zip(x, y, w):
        if ix > 0: 
            rightSum += iw
        elif ix < 0:
            leftSum += iw
    if leftSum > rightSum:
        ref_x = [-1.*rix for rix in rot_x]
        ref_y = rot_y
    else:
        ref_x = rot_x
        ref_y = rot_y
    
    return np.array(ref_x), np.array(ref_y)

def delta_phi(phi1, phi2):
  PI = 3.14159265359
  x = phi1 - phi2
  while x >=  PI:
      x -= ( 2*PI )
  while x <  -PI:
      x += ( 2*PI )
  return x

import numpy as np

import pandas

num_test = 3000

input_filename = "train.h5"
training_set = pandas.HDFStore(input_filename)
training_set = training_set.select("table",stop = 3*num_test)

top_training_set = training_set[training_set["is_signal_new"] == 1][:num_test]
qcd_training_set = training_set[training_set["is_signal_new"] == 0][:num_test]

top_training_set = np.array(top_training_set.iloc[:, :800])
qcd_training_set = np.array(qcd_training_set.iloc[:,:800])

import numpy as np

num_files = num_test
A_all = np.zeros((num_files,800))
B_all = np.zeros((num_files,800))

for j in range(num_files):
    #Reading the 100 files   
    A_all[j] = top_training_set[j]
    B_all[j] = qcd_training_set[j]

# import pandas
# input_filename = "train.h5"
# training_set = pandas.HDFStore(input_filename)



top_training_set_E = top_training_set[:,0::4]
top_training_set_px = top_training_set[:,1::4]
top_training_set_py = top_training_set[:,2::4,]
top_training_set_pz = top_training_set[:,3::4]

p = (top_training_set_px**2+top_training_set_py**2+top_training_set_pz**2) ** 0.5

eta = 0.5 * (np.log(p + top_training_set_pz ) - np.log(p - top_training_set_pz )) #pseudorapidity eta
eta = np.nan_to_num(eta)
phi = np.arctan2(top_training_set_py, top_training_set_px)  

def last_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)

for i in range(10):
    eta_value = eta[i]
    phi_value = phi[i]

eta.shape

dt = 0.05
left_x = -2
right_x = 2
left_y = -2
right_y = 2
xedges = np.linspace(left_x,right_x,(right_x-left_x)/dt+1)
yedges = np.linspace(left_y,right_y,(right_y-left_y)/dt+1)

H,xedges,yedges = np.histogram2d(eta[0][:23],phi[0][:23],bins = [xedges, yedges])

H_ave = np.zeros((H.shape[0],H.shape[1]))
for i in range(len(eta)):
    eta_i = eta[i]
    phi_i = phi[i]
    H,xedges,yedges = np.histogram2d(eta_i,phi_i,bins = [xedges, yedges])
    H_ave += H/np.sum(H)
H_ave = H_ave/len(eta)

ind = np.unravel_index(np.argmax(H_ave, axis=None), H_ave.shape)

H_ave[ind] = 0

np.max(H_ave)

top_training_set_E = top_training_set[:,0::4]
top_training_set_px = top_training_set[:,1::4]
top_training_set_py = top_training_set[:,2::4,]
top_training_set_pz = top_training_set[:,3::4]

p = (top_training_set_px**2+top_training_set_py**2+top_training_set_pz**2) ** 0.5

eta = 0.5 * (np.log(p + top_training_set_pz ) - np.log(p - top_training_set_pz )) #pseudorapidity eta
eta = np.nan_to_num(eta)
phi = np.arctan2(top_training_set_py, top_training_set_px)  

particlePt = np.sqrt(top_training_set_px**2 + top_training_set_py**2)
particleEta = eta
particlePhi = phi
top_Eta_nosym = eta
top_Phi_nosym = phi

maxPtIndex = np.argmax(particlePt,axis = 1)
x = np.zeros((particleEta.shape[0],particleEta.shape[1]))
for i in range(x.shape[0]):
    x[i] = particleEta[i,:] - particleEta[i,maxPtIndex[0]]
delta_phi_func = np.vectorize(delta_phi)
y = delta_phi_func(np.array(particlePhi), particlePhi[maxPtIndex])
w = np.array(particlePt)
#x, y = rotate_and_reflect(x[0], y[0], w[0])

symmetrized_eta = np.zeros((eta.shape[0],eta.shape[1]))
symmetrized_phi = np.zeros((eta.shape[0],eta.shape[1]))

for i in range(eta.shape[0]):
    symmetrized_eta[i], symmetrized_phi[i] = rotate_and_reflect(x[i],y[i],w[i])
    #symmetrized_phi[i] = rotate_and_reflect(x[i],y[i],w[i])

symmetrized_eta_top = np.nan_to_num(symmetrized_eta)
symmetrized_phi_top = np.nan_to_num(symmetrized_phi)

symmetrized_eta_top.shape

# dt = 0.1
# left_x = -1.5
# right_x = 1.5
# left_y = -2
# right_y = 2
# xedges = np.linspace(left_x,right_x,(right_x-left_x)/dt+1)
# yedges = np.linspace(left_y,right_y,(right_y-left_y)/dt+1)


H,xedges,yedges = np.histogram2d(symmetrized_eta_top[0],symmetrized_phi_top[0],bins = [xedges, yedges])

H_ave_sym = np.zeros((H.shape[0],H.shape[1]))
for i in range(len(eta)):
    eta_i = symmetrized_eta_top[i]
    phi_i = symmetrized_phi_top[i]
    H,xedges,yedges = np.histogram2d(eta_i,phi_i,bins = [xedges, yedges])
    H_ave_sym += H/np.sum(H)
H_ave_sym = H_ave_sym/len(eta)

ind = np.unravel_index(np.argmax(H_ave_sym, axis=None), H_ave_sym.shape)

H_ave_sym[ind] = 0

H_ave_sym

import matplotlib.pyplot as plt

plt.imshow(H_ave_sym, interpolation = 'nearest', origin='low', extent = [left_x,right_x,left_y,right_y])

#plt.imshow(H[0])
plt.show()

plt.imshow(H_ave, interpolation = 'nearest', origin='low', extent = [left_x,right_x,left_y,right_y])
plt.show()

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# QCD analysis

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

qcd_training_set_E = qcd_training_set[:,0::4]
qcd_training_set_px = qcd_training_set[:,1::4]
qcd_training_set_py = qcd_training_set[:,2::4,]
qcd_training_set_pz = qcd_training_set[:,3::4]

# training_set_E = qcd_training_set[:,0::4]
# training_set_px = qcd_training_set[:,1::4]
# training_set_py = qcd_training_set[:,2::4]
# training_set_pz = qcd_training_set[:,3::4]

particlePt = np.sqrt(qcd_training_set_px**2 + qcd_training_set_px**2)


p = (qcd_training_set_px**2+qcd_training_set_py**2+qcd_training_set_pz**2) ** 0.5

eta = 0.5 * (np.log(p + qcd_training_set_pz ) - np.log(p - qcd_training_set_pz )) #pseudorapidity eta
eta = np.nan_to_num(eta)
phi = np.arctan2(qcd_training_set_py, qcd_training_set_px)  

particleEta = eta
particlePhi = phi

qcd_Eta_nosym = eta
qcd_Phi_nosym = phi

maxPtIndex = np.argmax(particlePt,axis = 1)
x = np.zeros((particleEta.shape[0],particleEta.shape[1]))
for i in range(x.shape[0]):
    x[i] = particleEta[i,:] - particleEta[i,maxPtIndex[0]]
delta_phi_func = np.vectorize(delta_phi)
y = delta_phi_func(np.array(particlePhi), particlePhi[maxPtIndex])
w = np.array(particlePt)
#x, y = rotate_and_reflect(x[0], y[0], w[0])

symmetrized_eta = np.zeros((eta.shape[0],eta.shape[1]))
symmetrized_phi = np.zeros((eta.shape[0],eta.shape[1]))

for i in range(eta.shape[0]):
    symmetrized_eta[i], symmetrized_phi[i] = rotate_and_reflect(x[i],y[i],w[i])
    #symmetrized_phi[i] = rotate_and_reflect(x[i],y[i],w[i])

symmetrized_eta_qcd = np.nan_to_num(symmetrized_eta)
symmetrized_phi_qcd = np.nan_to_num(symmetrized_phi)

eta.shape

H,xedges,yedges = np.histogram2d(eta[0],phi[0],bins = [xedges, yedges])
ind = np.unravel_index(np.argmax(H, axis=None), H.shape)
H[ind] = 0

H_ave = np.zeros((H.shape[0],H.shape[1]))
for i in range(len(eta)):
    eta_i = eta[i]
    phi_i = phi[i]
    H,xedges,yedges = np.histogram2d(eta_i,phi_i,bins = [xedges, yedges])
    H_ave += H/np.sum(H)
H_ave = H_ave/len(eta)

ind = np.unravel_index(np.argmax(H_ave, axis=None), H_ave.shape)

H_ave[ind] = 0

np.max(H_ave)

particlePt = np.sqrt(qcd_training_set_px**2 + qcd_training_set_py**2)
particleEta = eta
particlePhi = phi

maxPtIndex = np.argmax(particlePt,axis = 1)
x = np.zeros((particleEta.shape[0],particleEta.shape[1]))
for i in range(x.shape[0]):
    x[i] = particleEta[i,:] - particleEta[i,maxPtIndex[0]]
delta_phi_func = np.vectorize(delta_phi)
y = delta_phi_func(np.array(particlePhi), particlePhi[maxPtIndex])
w = np.array(particlePt)
#x, y = rotate_and_reflect(x[0], y[0], w[0])

symmetrized_eta = np.zeros((eta.shape[0],eta.shape[1]))
symmetrized_phi = np.zeros((eta.shape[0],eta.shape[1]))

for i in range(eta.shape[0]):
    symmetrized_eta[i], symmetrized_phi[i] = rotate_and_reflect(x[i],y[i],w[i])
    #symmetrized_phi[i] = rotate_and_reflect(x[i],y[i],w[i])

symmetrized_eta_qcd = np.nan_to_num(symmetrized_eta_qcd)
symmetrized_phi_qcd = np.nan_to_num(symmetrized_phi_qcd)

symmetrized_eta_qcd.shape

H,xedges,yedges = np.histogram2d(symmetrized_eta_qcd[0],symmetrized_phi_qcd[0],bins = [xedges, yedges])

H_ave_sym = np.zeros((H.shape[0],H.shape[1]))
for i in range(len(eta)):
    eta_i = symmetrized_eta_qcd[i]
    phi_i = symmetrized_phi_qcd[i]
    H,xedges,yedges = np.histogram2d(eta_i,phi_i,bins = [xedges, yedges])
    if (np.sum(H)!= 0):
        H_ave_sym += H/np.sum(H)
H_ave_sym = H_ave_sym/len(eta)

ind = np.unravel_index(np.argmax(H_ave_sym, axis=None), H_ave_sym.shape)

H_ave_sym[ind] = 0

list(H_ave_sym)

plt.imshow(H_ave_sym, interpolation = 'nearest', origin='low', extent = [left_x,right_x,left_y,right_y])

#plt.imshow(H[0])
plt.show()

plt.imshow(H_ave, interpolation = 'nearest', origin='low', extent = [left_x,right_x,left_y,right_y])
plt.show()

## MANIFOLD LEARNING PART

#ONLY DO A FRACTION
sample_size = num_files


top_data =  np.zeros((sample_size,400))
qcd_data = np.zeros((sample_size,400))
top_data[:,:200] = symmetrized_eta_top
top_data[:,200:] = symmetrized_phi_top
qcd_data[:,:200] = symmetrized_eta_qcd
qcd_data[:,200:] = symmetrized_phi_qcd

from sklearn.preprocessing import scale

sample_size = int(sample_size/2)

X0 = np.zeros((2*sample_size,400))


Y0 = np.zeros(2*sample_size)
Y0[:sample_size] = 0
Y0[sample_size:] = 1

X0[:sample_size,:] = top_data[:sample_size]
X0[sample_size:,:] = qcd_data[:sample_size]

X0_s = X0.reshape((2*sample_size,400))
#X0_s = scale(X0)

**Symmetrized Jet** :Manifold Learning Jet Image $(\eta,\phi)$ 

from time import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

from sklearn import manifold, datasets

# Next line to silence pyflakes. This import is needed.
Axes3D


X = X0_s
color = Y0

n_neighbors = 10
n_components = 2

fig = plt.figure(figsize=(15, 8))
plt.suptitle("Manifold Learning with %i points, %i neighbors"
             % (1000, n_neighbors), fontsize=14)


ax = fig.add_subplot(251)
plt.scatter(X[:, 0], X[:, 1], c=color, cmap=plt.cm.Spectral,marker = 1)

methods = ['standard', 'ltsa', 'hessian', 'modified']
labels = ['LLE', 'LTSA', 'Hessian LLE', 'Modified LLE']

for i, method in enumerate(methods):
    try:
        t0 = time()
        Y = manifold.LocallyLinearEmbedding(n_neighbors, n_components,
                                            eigen_solver='auto',
                                            method=method).fit_transform(X)
        t1 = time()
        print("%s: %.2g sec" % (methods[i], t1 - t0))

        ax = fig.add_subplot(252 + i)
        plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral,marker = 1)
        plt.title("%s (%.2g sec)" % (labels[i], t1 - t0))
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        plt.axis('tight')
    except:
        continue

t0 = time()
Y = manifold.Isomap(n_neighbors, n_components).fit_transform(X)
t1 = time()
print("Isomap: %.2g sec" % (t1 - t0))
ax = fig.add_subplot(257)
plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral, marker = 1)
print("Accuracy for ISOMAP",  "is", cal_accuracy(Y,Y0,5))
plt.title("Isomap (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')


# t0 = time()
# mds = manifold.MDS(n_components, max_iter=100, n_init=1)
# Y = mds.fit_transform(X)
# t1 = time()
# print("MDS: %.2g sec" % (t1 - t0))
# ax = fig.add_subplot(258)
# plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral, marker = 1)
# plt.title("MDS (%.2g sec)" % (t1 - t0))
# ax.xaxis.set_major_formatter(NullFormatter())
# ax.yaxis.set_major_formatter(NullFormatter())
# plt.axis('tight')


# t0 = time()
# se = manifold.SpectralEmbedding(n_components=n_components,
#                                 n_neighbors=n_neighbors)
# Y = se.fit_transform(X)
# t1 = time()
# print("SpectralEmbedding: %.2g sec" % (t1 - t0))
# ax = fig.add_subplot(259)
# plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral, marker = 1)
# plt.title("SpectralEmbedding (%.2g sec)" % (t1 - t0))
# ax.xaxis.set_major_formatter(NullFormatter())
# ax.yaxis.set_major_formatter(NullFormatter())
# plt.axis('tight')

# t0 = time()
# tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
# Y = tsne.fit_transform(X)
# t1 = time()
# print("t-SNE: %.2g sec" % (t1 - t0))
# ax = fig.add_subplot(2, 5, 10)
# plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral, marker = 1)
# plt.title("t-SNE (%.2g sec)" % (t1 - t0))
# ax.xaxis.set_major_formatter(NullFormatter())
# ax.yaxis.set_major_formatter(NullFormatter())
# plt.axis('tight')

plt.show()

import pickle

# open a file, where you ant to store the data


saved_train = [X0,Y0,Y]
file = open('ML_train', 'wb')

# dump information to that file
pickle.dump(saved_train, file)

# close the file
file.close()





#For each point in X, find its 5NN
X0 = np.zeros((2*num_files,400))


Y0_test = np.zeros(2*num_files)
Y0_test[:int(num_files/2)] = 0
Y0_test[int(num_files/2):num_files] = 1
Y0_test[num_files:3*int(num_files/2)] = 0
Y0_test[3*int(num_files/2):] = 1

X0[:num_files,:] = top_data
X0[num_files:,:] = qcd_data

X0_test = X0.reshape((2*num_files,400))
k = 5

nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X0_test)
distances, indices = nbrs.kneighbors(X0_test)





X0.shape,Y0.shape,Y.shape,X0_test.shape, len(labels),color

indices

sample_size

#Corresponding y 
#for each blue in x -> plot blue
top_predicted_list  = []
qcd_predicted_list = []
k = 5
for i in range(len(Y)):
    for j in range(k):
        if (indices[i][j] < 3000):
            if (Y0[indices[i][j]] == 0):
                top_predicted_list.append(Y[indices[i][j]])
                #plot Y[indices[i][j]] blue
            if (Y0[indices[i][j]] == 1):
                qcd_predicted_list.append(Y[indices[i][j]])
                #plot Y[indices[i][j]] red



top_predicted_list = np.array(top_predicted_list)
qcd_predicted_list = np.array(qcd_predicted_list)

Y0.shape

top_predicted_list.shape

qcd_predicted_list.shape

len(color)

predicted_list = np.vstack((top_predicted_list,qcd_predicted_list))

color = np.zeros(len(predicted_list))
color[:len(top_predicted_list)] = 1



plt.scatter(predicted_list[:,0],predicted_list[:,1], c=color, cmap=plt.cm.Spectral, marker = 1)

from sklearn.neighbors import NearestNeighbors
#Calculate the percentage of 5 knn that is correctly clustered
def cal_accuracy(Y,labels,k):
    #find 5 -knn
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(Y)
    distances, indices = nbrs.kneighbors(Y)
    count = 0
    for i in range(len(Y)):
        for j in range(k):
            if (labels[i] ==labels[indices[i][j]]):
                count += 1
    return count/(k*len(Y))

sample_size = num_files
top_data =  np.zeros((sample_size,400))
qcd_data = np.zeros((sample_size,400))
top_data[:,:200] = top_Eta_nosym
top_data[:,200:] = top_Phi_nosym
qcd_data[:,:200] = qcd_Eta_nosym
qcd_data[:,200:] = qcd_Phi_nosym
from sklearn.preprocessing import scale


X0 = np.zeros((2*sample_size,400))


Y0 = np.zeros(2*num_files)
Y0[:sample_size] = 0
Y0[sample_size:] = 1

X0[:sample_size,:] = top_data
X0[sample_size:,:] = qcd_data

X0_s = X0.reshape((2*num_files,400))
#X0_s = scale(X0)

**Non Symmetrized Jet** (For comparison)

from time import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

from sklearn import manifold, datasets

# Next line to silence pyflakes. This import is needed.
Axes3D


X = X0_s
color = Y0

n_neighbors = 10
n_components = 2

fig = plt.figure(figsize=(15, 8))
plt.suptitle("Manifold Learning with %i points, %i neighbors"
             % (1000, n_neighbors), fontsize=14)


ax = fig.add_subplot(251)
plt.scatter(X[:, 0], X[:, 1], c=color, cmap=plt.cm.Spectral,marker = 1)

methods = ['standard', 'ltsa', 'hessian', 'modified']
labels = ['LLE', 'LTSA', 'Hessian LLE', 'Modified LLE']

for i, method in enumerate(methods):
    try:
        t0 = time()
        Y = manifold.LocallyLinearEmbedding(n_neighbors, n_components,
                                            eigen_solver='auto',
                                            method=method).fit_transform(X)
        t1 = time()
        print("%s: %.2g sec" % (methods[i], t1 - t0))

        ax = fig.add_subplot(252 + i)
        plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral,marker = 1)
        plt.title("%s (%.2g sec)" % (labels[i], t1 - t0))
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        plt.axis('tight')
    except:
        continue

t0 = time()
Y = manifold.Isomap(n_neighbors, n_components).fit_transform(X)
t1 = time()
print("Isomap: %.2g sec" % (t1 - t0))
ax = fig.add_subplot(257)
plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral, marker = 1)
print("Accuracy for ISOMAP",  "is", cal_accuracy(Y,Y0,5))
plt.title("Isomap (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')


# t0 = time()
# mds = manifold.MDS(n_components, max_iter=100, n_init=1)
# Y = mds.fit_transform(X)
# t1 = time()
# print("MDS: %.2g sec" % (t1 - t0))
# ax = fig.add_subplot(258)
# plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral, marker = 1)
# plt.title("MDS (%.2g sec)" % (t1 - t0))
# ax.xaxis.set_major_formatter(NullFormatter())
# ax.yaxis.set_major_formatter(NullFormatter())
# plt.axis('tight')


t0 = time()
se = manifold.SpectralEmbedding(n_components=n_components,
                                n_neighbors=n_neighbors)
Y = se.fit_transform(X)
t1 = time()
print("SpectralEmbedding: %.2g sec" % (t1 - t0))
ax = fig.add_subplot(259)
plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral, marker = 1)
plt.title("SpectralEmbedding (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')

# t0 = time()
# tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
# Y = tsne.fit_transform(X)
# t1 = time()
# print("t-SNE: %.2g sec" % (t1 - t0))
# ax = fig.add_subplot(2, 5, 10)
# plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral, marker = 1)
# plt.title("t-SNE (%.2g sec)" % (t1 - t0))
# ax.xaxis.set_major_formatter(NullFormatter())
# ax.yaxis.set_major_formatter(NullFormatter())
# plt.axis('tight')

plt.show()