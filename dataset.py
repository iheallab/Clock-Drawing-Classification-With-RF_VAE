import os
import torch
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def to_rgb(array):
    empty_array = np.empty((array.shape[0], 3, 64, 64))
    for j in range(len(array)):
        row = np.stack((array[j],array[j], array[j]), axis=0)
        empty_array[j] = row
    return empty_array

# Loads .png images from a directory into a torch tensor. Aligns demographics tensor:
def images_to_torch(image_path, demo_names = None):
    num_images = 0
    if demo_names is None:
        with os.scandir(image_path) as files:
            for file in files:
                num_images+=1
    else:
        num_images = len(demo_names)
    demographics = np.empty((num_images, 4))
    images = np.empty((num_images, 64, 64))
    i = 0
    with os.scandir(image_path) as files:
      # loops through each file in the directory
        for file in files:
            if file.name.endswith('.png'):

                # only do clocks that have demographics if demographics are used
                if demo_names is not None:
                    if file.name not in demo_names:
                        continue
                    else:
                        demographics[i] = demo_names[file.name]

              # adds only the image files to the clock list
                img = Image.open(image_path + file.name)
                array = np.array(img)
                images[i] = array
                i+=1
    images = to_rgb(images)
    
    images = np.reshape(images, (images.shape[0], 3, 64, 64))
    images = torch.from_numpy(images)
    return images, torch.from_numpy(demographics)

# load demographics from csv file:
def loadDemographics(path):
    demo_dict = {}
    with open(path) as demo_read:
        lines = demo_read.readlines()
        for i in range(len(lines)):
            line = lines[i].split(',')
            demo_dict[line[0]] = line[1:]
    return demo_dict

def createDatasets(dementia_path, control_path, useDemographics=False, dementia_demo_path="", control_demo_path=""):
    print("Preparing datasets...")
    dementia_names = None
    control_names = None
    
    # Get demographics with their filenames to allow matching them together:
    if useDemographics:
        dementia_names = loadDemographics(dementia_demo_path)
        control_names = loadDemographics(control_demo_path)
    
    # Convert images to torch tensors. Create matching demographic dataset:
    dementia_x, dementia_demo = images_to_torch(dementia_path, dementia_names)
    control_x, control_demo = images_to_torch(control_path, control_names)

    control_y = torch.zeros(len(control_x), 1)
    dementia_y = torch.ones(len(dementia_x), 1)
           
    all_y = torch.cat((dementia_y, control_y)).numpy()
    all_x = torch.cat((dementia_x, control_x)).numpy()
    all_x = all_x/255

    # Squeezes array, converts it to torch tensor, makes it float tensor, adds it to device:
    def finalizeArray(array):
        array = array.squeeze()
        array = torch.from_numpy(array)
        array = array.type(torch.FloatTensor)
        array = array.to(device)
        return array

    # Create train and test sets:
    sss = StratifiedShuffleSplit(train_size=0.75, test_size=0.25, random_state=25)
    sss.get_n_splits(all_x, all_y)
    for train_index, test_index in sss.split(all_x, all_y):
        train_x, test_x = all_x[train_index], all_x[test_index]
        train_y, test_y = all_y[train_index], all_y[test_index]

        if useDemographics:
            all_demo = torch.cat((dementia_demo, control_demo)).numpy()
            train_demo, test_demo = all_demo[train_index], all_demo[test_index]
            train_demo, test_demo = finalizeArray(train_demo), finalizeArray(test_demo)
        else:
            train_demo, test_demo = None, None
        
    datasets = [train_x, train_y, test_x, test_y]
    for i in range(len(datasets)):
        datasets[i] = finalizeArray(datasets[i])
    train_x, train_y, test_x, test_y = datasets[0], datasets[1], datasets[2], datasets[3]
    
    return train_x, train_y, test_x, test_y, train_demo, test_demo