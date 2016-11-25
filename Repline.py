'''
Created on Nov 25, 2016

@author: Caleb Hulbert
'''

from Tkinter import Tk
import tkFileDialog
import os


class Repline(object):
    '''
    This Class holds one Line from one Rep of Corn Cobs, each with  one or more corn cobs,
    and each corn Cob with one or more Kernel.
    '''

    def __init__(self, startInDirectory='', row=''):
        '''
        '''
        Tk.withdraw()
        self.directory = ''
        self.setdirectoryandrow(startInDirectory, row)
        self.setclusters()
        self.setstats()

    def setdirectoryandrow(self, startindirectory='', row=''):
        '''
        ask user for row name and directory to find row files, unless they are given as an argument.
        '''
        if startindirectory == '':
            self.directory = str(tkFileDialog.askdirectory())
        else:
            self.directory = os.path.abspath(startindirectory)
        if row == '':
            filename = str(tkFileDialog.askopenfile(initialdir=self.directory).name)
            self.name = filename[len(self.directory) + 3:-8]
