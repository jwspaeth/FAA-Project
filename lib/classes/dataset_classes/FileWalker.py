import os
from lib.classes.dataset_classes.DataProcessor import data_processor


def FileWalker(directory_path):
    master_array = []
    for subdir, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = subdir + os.sep + file
            master_array.append(data_processor(file_path))

    return master_array
