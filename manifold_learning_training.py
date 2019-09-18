import numpy as np
import pandas
from sklearn.neighbors import NearestNeighbors
import pickle
import numpy as np
import pandas
from time import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

from sklearn import manifold, datasets
from sklearn.preprocessing import scale
import logging

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

def last_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)




def step2():
    num_test = 5000
    
    sample_index = 1
    # open a file, where you stored the pickled data
    file = open('sample_data/sample_data_training_top'+str(sample_index), 'rb')

    # dump information to that file
    top_training_set = pickle.load(file)
    top_training_set = top_training_set.sample(n=num_test)

    file.close()

    # open a file, where you stored the pickled data
    file = open('sample_data/sample_data_training_qcd'+str(sample_index), 'rb')

    # dump information to that file
    qcd_training_set = pickle.load(file)
    qcd_training_set = qcd_training_set.sample(n=num_test)

    file.close()


    # TOP
    ##################################
    top_training_set = np.array(top_training_set.iloc[:, :800])
    qcd_training_set = np.array(qcd_training_set.iloc[:,:800])

    num_files = num_test
    A_all = np.zeros((num_files,800))
    B_all = np.zeros((num_files,800))

    for j in range(num_files):
        A_all[j] = top_training_set[j]
        B_all[j] = qcd_training_set[j]

    top_training_set_E = top_training_set[:,0::4]
    top_training_set_px = top_training_set[:,1::4]
    top_training_set_py = top_training_set[:,2::4,]
    top_training_set_pz = top_training_set[:,3::4]

    p = (top_training_set_px**2+top_training_set_py**2+top_training_set_pz**2) ** 0.5

    eta = 0.5 * (np.log(p + top_training_set_pz ) - np.log(p - top_training_set_pz )) #pseudorapidity eta
    eta = np.nan_to_num(eta)
    phi = np.arctan2(top_training_set_py, top_training_set_px)  

    for i in range(10):
        eta_value = eta[i]
        phi_value = phi[i]

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

    symmetrized_eta = np.zeros((eta.shape[0],eta.shape[1]))
    symmetrized_phi = np.zeros((eta.shape[0],eta.shape[1]))

    for i in range(eta.shape[0]):
        symmetrized_eta[i], symmetrized_phi[i] = rotate_and_reflect(x[i],y[i],w[i])
    logging.info("Finish symmetrize top")

    symmetrized_eta_top = np.nan_to_num(symmetrized_eta)
    symmetrized_phi_top = np.nan_to_num(symmetrized_phi)

    # QCD
    #####################################################
    qcd_training_set_E = qcd_training_set[:,0::4]
    qcd_training_set_px = qcd_training_set[:,1::4]
    qcd_training_set_py = qcd_training_set[:,2::4,]
    qcd_training_set_pz = qcd_training_set[:,3::4]

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

    symmetrized_eta_qcd = np.nan_to_num(symmetrized_eta)
    symmetrized_phi_qcd = np.nan_to_num(symmetrized_phi)
    logging.info("Finish symmetrize qcd")

    # Next line to silence pyflakes. This import is needed.
    #Axes3D

    sample_size = num_files


    top_data =  np.zeros((sample_size,400))
    qcd_data = np.zeros((sample_size,400))
    top_data[:,:200] = symmetrized_eta_top
    top_data[:,200:] = symmetrized_phi_top
    qcd_data[:,:200] = symmetrized_eta_qcd
    qcd_data[:,200:] = symmetrized_phi_qcd



    sample_size = int(sample_size/2)

    X0 = np.zeros((2*sample_size,400))


    Y0 = np.zeros(2*sample_size)
    Y0[:sample_size] = 0
    Y0[sample_size:] = 1

    X0[:sample_size,:] = top_data[:sample_size]
    X0[sample_size:,:] = qcd_data[:sample_size]

    X0_s = X0.reshape((2*sample_size,400))
    X0_s = scale(X0)
    X = X0_s
    color = Y0

    n_neighbors = 5
    n_components = 2

    logging.info("Start manifold learning")
    Y = manifold.Isomap(n_neighbors, n_components).fit_transform(X)
    logging.info("Finish manifold learning")

    saved_train = [X0,Y0,Y]
    file = open('ML_train'+str(sample_index)+"paper", 'wb')

    # dump information to that file
    pickle.dump(saved_train, file)

    # close the file
    file.close()