'''
Created on Sep 14, 2016

@author: Caleb Hulbert
'''
import Kernel
from sklearn.cluster import KMeans


class Cob(object):
    '''
    holds a list of Kernels, with associated statistics

    -name
    -kernellist         list of kernels
    -numkernels         int
    -cluster1           [size, center]
    -cluster2           [size, center]
    -mean
    '''

    def __init__(self, kernellist, name):
        '''
        creates the Cob.
        '''
        self.name = name
        self.kernellist = kernellist
        self.kernelcenters = []
        self.setkernelcenters()
        self.cluster1 = [0, [0, 0, 0]]
        self.cluster2 = [0, [0, 0, 0]]
        self.setclusters()
        self.setstas()

    def setkernelcenters(self):
        '''
        create a list of all the centers calculated for the kernels this Cob contains
        '''
        for kernel in self.kernellist:
            self.kernelcenters.append(kernel.cluster1[1])
            self.kernelcenters.append(kernel.cluster2[1])

    def setclusters(self):
        '''
        calculates kmeans centers for the cob from the centers of the kernels it contains
        '''
        kmeans = KMeans(n_clusters=1).fit(self.kernelcenters)
        self.cluster1[1] = kmeans.cluster_centers_[0]
        self.cluster1[1] = kmeans.cluster_centers_[1]
        sizec1 = 0
        sizec2 = 0
        for label in kmeans.labels_:
            if label == 0:
                sizec1 += 1
            elif label == 1:
                sizec2 += 1
        self.cluster1[0] = sizec1
        self.cluster2[0] = sizec2

    def setstats(self):
        L = 0
        a = 0
        b = 0
        for kernel in self.kernellist: 
            L += kernel.mean[0]
            