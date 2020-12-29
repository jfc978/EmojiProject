# -*- coding: utf-8 -*-
"""
Function to remove need to write code for accessing individual files post-parsing

@author: James
"""
import os
from os.path import dirname, abspath
import numpy as np

#define dataset object
class Data:
    def __init__(self):
        original_dir = "data\\original"
        parsed_dir = "data\\parsed"

        d = dirname(dirname(abspath(__file__))) # /../EmojiProject
    
        #data file directories
        data_files = os.path.join(d, original_dir)
        parsed_files = os.path.join(d, parsed_dir)
    
        #call parse function, passing directory or foreach file
        #TODO
        
        #create list from parsed file directories
        files = os.listdir(parsed_files)
        file_list = []
        for file in files:
            #open target file
            filepath = os.path.join(parsed_files, file)
            file_list.append(filepath)
        self.files = file_list
        
    #split dataset
    def split_data(self, ratio):
        self.training = []
        self.testing = []
        index = []
        training_num = round(ratio*len(self.files))-1
        
        #randomly select training_num files from file list, adding to index to filter
        for i in range(0,training_num):
            data_point = np.random.randint(0,len(self.files))
            while data_point in index:
                data_point = np.random.randint(0,len(self.files)) 
            index.append(data_point)
            self.training.append(self.files[data_point])
        
        #add all files not in training set to testing set
        for i in range(0,len(self.files)):
            if i not in index:
                self.testing.append(self.files[i])
        
        return self.training, self.testing
    
if __name__ == '__main__':
    data = Data()
    training, testing = data.split_data(0.7)