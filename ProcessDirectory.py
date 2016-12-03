'''
Created on Nov, 29 2016

@author: Caleb Hulbert
'''
from Tkinter import Tk
import tkFileDialog
import os

import Repline


def start(startindirectory='', write=True, clustertype='kmeans', stats='False'):
    Tk().withdraw()
    if startindirectory == '':
        startindirectory = str(tkFileDialog.askdirectory())
    print str(startindirectory)
    repnames = []
    for cobfile in os.listdir(startindirectory):
        if ".tif.csv" in cobfile and "1_" in cobfile:
            #Remove .csv
            repname = os.path.splitext(cobfile)[0]
            #Remove .tif
            repname = os.path.splitext(repname)[0]
            # remove leading number and underscore
            repname = repname[repname.find('_') + 1:]
            repnames.append(repname)
    allcolorinfo = [["Repline", "Pixels in Cluster", "L", "a", "b"]]
    print "Constructing Replines"
    for namenum in xrange(len(repnames)):
        repline = Repline.Repline(
            startInDirectory=startindirectory, row=repnames[namenum], clustertype=clustertype, stats=stats)
        for cluster in repline.clusters:
            colorinfo = [repnames[namenum], cluster[0], cluster[1][0], cluster[1][1], cluster[1][2]]
            allcolorinfo.append(colorinfo)
            # print colorinfo
        if namenum < len(repnames) - 1:
            print "(", namenum + 1, "/", len(repnames), ")", "Finished: ", repline.name, "Constructing ", repnames[namenum + 1], "..."
        elif namenum == len(repnames):
            print "(", namenum + 1, "/", len(repnames), ")", "Finished: ", repline.name
            print "Done!"
    newFile = str(startindirectory) + "/TotalStats.csv"
    with open(newFile, 'w') as results:
        for line in allcolorinfo:
            print line
            results.write("%s,%s,%s,%s,%s" % (line[0], line[1], line[2],line[3],line[4]))
            results.write("\n")
    return allcolorinfo

def test():
    start('C:/Users/cmhul/Google Drive/College_/Corn_Color_Phenotyping/Landrace_Colorimeter_and_Pictures/Kernel CSVs')

if __name__ == '__main__':
    start('C:/Users/cmhul/Google Drive/College_/Corn_Color_Phenotyping/Landrace_Colorimeter_and_Pictures/Kernel CSVs')
