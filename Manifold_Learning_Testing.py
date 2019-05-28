import pickle

# open a file, where you stored the pickled data
file = open('ML_train', 'rb')

# dump information to that file
data = pickle.load(file)

# close the file
file.close()


X0_train = data[0]
Y0_train = data[1]
Y_train = data[2]

import numpy as np

import pandas

num_files = 5000
num_test = num_files

input_filename = "test.h5"
testing_set = pandas.HDFStore(input_filename)
testing_set = testing_set.select("table",stop = 20*num_test)

top_testing_set = testing_set[testing_set["is_signal_new"] == 1][3*num_test:4*num_test]
qcd_testing_set = testing_set[testing_set["is_signal_new"] == 0][3*num_test:4*num_test]

top_testing_set = np.array(top_testing_set.iloc[:, :800])
qcd_testing_set = np.array(qcd_testing_set.iloc[:,:800])

top_testing_set_E = top_testing_set[:,0::4]
top_testing_set_px = top_testing_set[:,1::4]
top_testing_set_py = top_testing_set[:,2::4,]
top_testing_set_pz = top_testing_set[:,3::4]

p = (top_testing_set_px**2+top_testing_set_py**2+top_testing_set_pz**2) ** 0.5

eta = 0.5 * (np.log(p + top_testing_set_pz ) - np.log(p - top_testing_set_pz )) #pseudorapidity eta
eta = np.nan_to_num(eta)
phi = np.arctan2(top_testing_set_py, top_testing_set_px)  

particlePt = np.sqrt(top_testing_set_px**2 + top_testing_set_py**2)
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



qcd_testing_set_E = qcd_testing_set[:,0::4]
qcd_testing_set_px = qcd_testing_set[:,1::4]
qcd_testing_set_py = qcd_testing_set[:,2::4,]
qcd_testing_set_pz = qcd_testing_set[:,3::4]

# testing_set_E = qcd_testing_set[:,0::4]
# testing_set_px = qcd_testing_set[:,1::4]
# testing_set_py = qcd_testing_set[:,2::4]
# testing_set_pz = qcd_testing_set[:,3::4]

particlePt = np.sqrt(qcd_testing_set_px**2 + qcd_testing_set_py**2)


p = (qcd_testing_set_px**2+qcd_testing_set_py**2+qcd_testing_set_pz**2) ** 0.5

eta = 0.5 * (np.log(p + qcd_testing_set_pz ) - np.log(p - qcd_testing_set_pz )) #pseudorapidity eta
eta = np.nan_to_num(eta)
phi = np.arctan2(qcd_testing_set_py, qcd_testing_set_px)  

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




#ONLY DO A FRACTION
sample_size = num_files


top_data =  np.zeros((sample_size,400))
qcd_data = np.zeros((sample_size,400))
top_data[:,:200] = symmetrized_eta_top
top_data[:,200:] = symmetrized_phi_top
qcd_data[:,:200] = symmetrized_eta_qcd
qcd_data[:,200:] = symmetrized_phi_qcd

X0 = np.zeros((2*sample_size,400))


Y0 = np.zeros(2*sample_size)
Y0[:sample_size] = 0
Y0[sample_size:] = 1

X0[:sample_size,:] = top_data[:sample_size]
X0[sample_size:,:] = qcd_data[:sample_size]

X0 = X0.reshape((2*sample_size,400))
#X0_s = scale(X0)

from sklearn.neighbors import NearestNeighbors
#For each point in X, find its 5NN
X0_test = np.zeros((2*num_files,400))

#Never use this, pretend to not know



X0_test[:num_files,:] = top_data
X0_test[num_files:,:] = qcd_data

X0_test = X0_test.reshape((2*num_files,400))
k = 5

nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X0_train)
distances, indices = nbrs.kneighbors(X0_test)

#Corresponding y 
#for each blue in x -> plot blue
top_predicted_list  = []
qcd_predicted_list = []
k = 5
for i in range(len(X0_test)):
    for j in range(k):
        if (Y0_train[indices[i][j]] == 0):
            top_predicted_list.append(Y_train[indices[i][j]])
        if (Y0_train[indices[i][j]] == 1):
            qcd_predicted_list.append(Y_train[indices[i][j]])
top_predicted_list = np.array(top_predicted_list)
qcd_predicted_list = np.array(qcd_predicted_list)
predicted_list = np.vstack((top_predicted_list,qcd_predicted_list))
color = np.zeros(len(predicted_list))
color[:len(top_predicted_list)] = 0
color[len(top_predicted_list):] = 1
import matplotlib.pyplot as plt
plt.scatter(predicted_list[:,0],predicted_list[:,1], c=color, cmap=plt.cm.Spectral, marker = 1)
plt.show()

predicted_list = []
for i in range(len(X0_test)):
    predict_val = 0*Y_train[indices[i][0]]
    for j in range(k):
        predict_val += Y_train[indices[i][j]]
    predict_val = predict_val/k
    predicted_list.append(predict_val)

predicted_list = np.array(predicted_list)
predicted_list.shape

color = np.zeros(2*num_files)
color[:num_files] = 0
color[num_files:] = 1

import matplotlib.pyplot as plt

plt.scatter(predicted_list[:,0],predicted_list[:,1], c=color, cmap=plt.cm.Spectral, marker = 1)
plt.title("Testing Manifold Learning for 5000 data points (trial 4)")

plt.xlabel("x")
plt.ylabel("y")
plt.show()



import matplotlib.pyplot as plt

plt.scatter(predicted_list[:,0],predicted_list[:,1], c=color, cmap=plt.cm.Spectral, marker = 1)
plt.title("Testing Manifold Learning for 5000 data points (trial 1)")

plt.xlabel("x")
plt.ylabel("y")
plt.show()

import matplotlib.pyplot as plt

plt.scatter(predicted_list[:,0],predicted_list[:,1], c=color, cmap=plt.cm.Spectral, marker = 1)
plt.title("Testing Manifold Learning for 5000 data points (trial 2)")

plt.xlabel("x")
plt.ylabel("y")
plt.show()

import matplotlib.pyplot as plt

plt.scatter(predicted_list[:,0],predicted_list[:,1], c=color, cmap=plt.cm.Spectral, marker = 1)
plt.title("Testing Manifold Learning for 5000 data points (trial 3)")

plt.xlabel("x")
plt.ylabel("y")
plt.show()

import matplotlib.pyplot as plt
size = len(predicted_list)

plt.scatter(predicted_list[:,0],predicted_list[:,1], c=color, cmap=plt.cm.Spectral, marker = 1)
plt.title("Testing Manifold Learning for 100000 data points")

plt.xlabel("x")
plt.ylabel("y")
plt.show()

top_predicted_list.shape,qcd_predicted_list.shape

Y0_train

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

#Accuracy Testing

Y0_test = np.zeros(2*num_files)
Y0_test[:int(num_files/2)] = 0
Y0_test[int(num_files/2):num_files] = 0
Y0_test[num_files:3*int(num_files/2)] = 1
Y0_test[3*int(num_files/2):] = 1