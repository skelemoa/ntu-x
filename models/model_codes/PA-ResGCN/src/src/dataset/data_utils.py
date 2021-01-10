import numpy as np


def multi_input(data, conn):
    C, T, V, M = data.shape
    data_new = np.zeros((3, C*2, T, V, M))
    data_new[0,:C,:,:,:] = data
    for i in range(V):
        data_new[0,C:,:,i,:] = data[:,:,i,:] - data[:,:,1,:]
    for i in range(T-2):
        data_new[1,:C,i,:,:] = data[:,i+1,:,:] - data[:,i,:,:]
        data_new[1,C:,i,:,:] = data[:,i+2,:,:] - data[:,i,:,:]
    for i in range(len(conn)):
        data_new[2,:C,:,i,:] = data[:,:,i,:] - data[:,:,conn[i],:]
    bone_length = 0
    for i in range(C):
        bone_length += np.power(data_new[2,i,:,:,:], 2)
    bone_length = np.sqrt(bone_length) + 0.0001
    for i in range(C):
        data_new[2,C+i,:,:,:] = np.arccos(data_new[2,i,:,:,:] / bone_length)
    return data_new
