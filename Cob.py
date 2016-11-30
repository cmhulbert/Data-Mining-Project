'''
Created on Sep 14, 2016

@author: Caleb Hulbert
'''
import Kernel
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Cob(object):
    '''
    holds a list of Kernels, with associated statistics

    -name
    -kernellist         list of kernels
    -numkernels         int
    -mode               pixels or kernels
    -cluster1           [size, center]
    -cluster2           [size, center]
    -mean
    '''

    def __init__(self, kernellist, name, clustertype="kmeans", pixelcluster=True, stats=False):
        self.name = name
        self.clustertype = clustertype
        self.kernellist = kernellist
        self.numkernels = len(self.kernellist)
        self.type = 'pixels'
        self.mean = [0, 0, 0]
        if stats == True:
            self.setstats()
        if pixelcluster == False:
            self.type = 'kernels'
            self.cluster1 = [0, [0, 0, 0]]
            self.cluster2 = [0, [0, 0, 0]]
            self.kernelcenters = []
            self.setkernelcenters()
        elif self.clustertype == "kmeans":
            self.setclusters(pixelcluster=pixelcluster)
            self.mean = [0, 0, 0]
        elif self.clustertype == "dbscan":
            self.clusters = []
            self.dbscan()

    def setkernelcenters(self):
        '''
        create a list of all the centers calculated for the kernels this Cob contains
        '''
        for kernel in self.kernellist:
            self.kernelcenters.append(kernel.cluster1[1])
            self.kernelcenters.append(kernel.cluster2[1])

    def setclusters(self, pixelcluster=True):
        '''
        calculates kmeans centers for the cob from the centers of the kernels it contains
        '''

        pixellist = []
        for kernel in self.kernellist:
            pixellist.extend(kernel.pixellist)
        kmeans = KMeans(n_clusters=2).fit(pixellist)
        self.cluster1[1] = kmeans.cluster_centers_[0].tolist()
        self.cluster2[1] = kmeans.cluster_centers_[1].tolist()
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
        '''
        calculates the mean in [L,a,b] form and set the the Cob.mean property
        '''
        L = 0
        a = 0
        b = 0
        numkernels = 0
        for kernel in self.kernellist:
            L += kernel.mean[0]
            a += kernel.mean[1]
            b += kernel.mean[2]
            numkernels += 1
        self.mean[0] = L / float(numkernels)
        self.mean[1] = a / float(numkernels)
        self.mean[2] = b / float(numkernels)

    def dbscan(self, eps=.5, plot=False):
        pixellist = []
        for kernel in self.kernellist:
            pixellist.extend(kernel.pixellist)
        X = StandardScaler().fit_transform(pixellist)
        density = eps
        db = DBSCAN(eps=density).fit(pixellist)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        print 'Estimated number of clusters: %d' % n_clusters_
        unique_labels = set(labels)
        colors = plt.cm.get_cmap('Spectral')(
            np.linspace(0, 1, len(unique_labels)))
        for k, col in zip(unique_labels, colors):
            if k == -1:
                col = 'k'
            class_member_mask = (labels == k)
            xyz = X[class_member_mask & core_samples_mask]
            llist, alist, blist = xyz[:, 0], xyz[:, 1], xyz[:, 2]
            if len(llist) > 0:
                lmean = llist.mean()
                amean = alist.mean()
                bmean = blist.mean()
                self.clusters.append([len(llist),[lmean,amean,bmean]])
        if plot is True:
            graph = plt.figure()
            ax = graph.add_subplot(111, projection='3d')
            for k, col in zip(unique_labels, colors):
                if k == -1:
                    col = 'k'
                class_member_mask = (labels == k)
                xyz = X[class_member_mask & core_samples_mask]
                ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=col)
                xyz = X[class_member_mask & ~core_samples_mask]
                ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=col, marker='.')
            ax.set_xlabel('L')
            ax.set_ylabel('a')
            ax.set_zlabel('b')
            plt.title('Estimated number of clusters: %d' % n_clusters_)
            plt.ion()
            plt.show()
        return db

    def showscatterplot(self):
        '''
        creates a 3d scatter plot whose points are either pixels from all the kernels in this cob,
        or centers from the kmean clusters of each kernel in this cob.
        '''
        if self.mode == 'pixels':
            pixels = []
            for kernel in self.kernellist:
                pixels.extend(kernel.pixellist)
            lablists = Kernel.threeTupleToThreeLists(pixels)
        else:
            kernelcenters = []
            for kernel in self.kernellist:
                kernelcenters.append(kernel.cluster1[1])
                kernelcenters.append(kernel.cluster2[1])
            lablists = Kernel.threeTupleToThreeLists(kernelcenters)
        plot = plt.figure()
        plt.close(1)
        del plot
        plot = plt.figure()
        axes = plot.add_subplot(111, projection='3d')
        axes.scatter(lablists[0], lablists[1], lablists[2], c='b', marker='.')
        axes.set_xlabel('L')
        axes.set_ylabel('a')
        axes.set_zlabel('b')
        if self.cluster1[1] != self.cluster2[1]:
            Kernel.addpoints(
                [self.cluster1[1], self.cluster2[1]], axes, marker='o')
        elif self.mean != [0, 0, 0]:
            Kernel.addpoints(self.mean, axes, color='g', marker='o')
        plt.ion()
        plt.show()
        return axes
