import pandas as pd
from keras.preprocessing import image
import numpy as np
from PIL import Image, ImageFilter
import os
import h5py
from keras.applications.vgg16 import preprocess_input
import matplotlib.image as mpimg

# Function to load images
def load_gt_hdf5(img_names, hdf5_dataset, hdf5_attr, path, shape):
    for i in range(len(img_names)):
        img = image.load_img(os.path.join(path, img_names[i])).resize(shape)
        hdf5_dataset[hdf5_attr][i, ...]= np.expand_dims(np.asarray(img, dtype=np.float32), axis=2)
    print(hdf5_attr, ' loaded.')

def load_images_hdf5(img_names, hdf5_dataset, hdf5_attr, path, shape):
    for i in range(len(img_names)):
        img = image.load_img(os.path.join(path, img_names[i])).resize(shape).convert('RGB')
        hdf5_dataset[hdf5_attr][i, ...]= np.asarray(img, dtype=np.float32)
    print(hdf5_attr, ' loaded.')

# Function to load counts
def load_data(data, hdf5_dataset, hdf5_attr):
    for i in range(len(data)):
        hdf5_dataset[hdf5_attr][i] = data[i]
    print(hdf5_attr, ' loaded.')
    
def load_images_PIL(img_names, file_path='./', preprocess='image', target_size=None):  
    if preprocess == 'image':
        images = []
        for im in img_names:
            img = mpimg.imread(os.path.join(file_path, im), target_size=(224, 224))
            img = np.expand_dims(img, axis=2)
            img = np.expand_dims(img, axis=0)
            images.append(img)
        return np.asarray(images)
    # Ground truth processing
    elif preprocess == 'gt':
        images = []
        for im in img_names:
            img = Image.open(os.path.join(file_path, im))
            img = img.resize(target_size, Image.ANTIALIAS).convert('1')
            img = np.asarray(img, dtype=np.float32)
            img = np.expand_dims(img, axis=2)
            images.append(img)
        return np.asarray(images)
    
def load_images(img_names, file_path='./', preprocess='image', target_size=None):  
    if preprocess == 'image':
        images = []
        for im in img_names:
            img = image.load_img(os.path.join(file_path, im), target_size=target_size)
            img = image.img_to_array(img)
            #img = np.expand_dims(img, axis=0)
            images.append(img)
        return np.asarray(images)
    # Ground truth processing
    elif preprocess == 'gt':
        images = []
        for im in img_names:
            img = Image.open(os.path.join(file_path, im))
            img = img.resize(target_size, Image.ANTIALIAS).convert('1')
            img = np.asarray(img, dtype=np.float32)
            img = np.expand_dims(img, axis=2)
            images.append(img)
        return np.asarray(images)
    
def create_dataset(hdf5_filename):
    trn_mta = pd.read_csv('C:\\Users\\Nolan\\Projects\\CSC590-CellCounting\\CellCounter\\training_set.csv')
    val_mta = pd.read_csv('C:\\Users\\Nolan\\Projects\\CSC590-CellCounting\\CellCounter\\validation_set.csv')
    ground_truth_path = 'C:\\Users\\Nolan\\Projects\\CSC590-CellCounting\\TrainingData\\BBBC005_v1_ground_truth'
    images_path = 'C:\\Users\\Nolan\\Projects\\CSC590-CellCounting\\TrainingData\\BBBC005_v1_images\\BBBC005_v1_images'

    # creates a shape which is (960, 520, 696, 3)
    trn_shp = (trn_mta.shape[0], 520, 696, 3)
    trn_shp_y = (trn_mta.shape[0], 512, 672, 1)
    val_shp = (val_mta.shape[0], 520, 696, 3)
    val_shp_y = (trn_mta.shape[0], 512, 672, 1)
    print('Dataset shapes: ', trn_shp, ' ', val_shp)

    # Now that I have shapes, create the hdf file with datasets
    #%%
    dt = h5py.special_dtype(vlen=str) # A custom dataset for file strings
    hdf5_file = h5py.File(hdf5_filename, mode='w')
    hdf5_file.create_dataset('train_img', trn_shp, np.float32)
    hdf5_file.create_dataset('train_counts', (trn_mta.shape[0],), np.int8)
    hdf5_file.create_dataset('train_gt', trn_shp_y, np.float32,)
    hdf5_file.create_dataset('train_labels', (trn_mta.shape[0],), dt)

    hdf5_file.create_dataset('val_img', val_shp, np.float32)
    hdf5_file.create_dataset('val_counts', (val_mta.shape[0],), np.int8)
    hdf5_file.create_dataset('val_gt', val_shp_y, np.float32)
    hdf5_file.create_dataset('val_labels', (val_mta.shape[0],), dt)
    # Load training data
    load_data(trn_mta.filename, hdf5_file, 'train_labels')
    load_data(trn_mta['count'], hdf5_file, 'train_counts')
    load_images(trn_mta.filename, hdf5_file, 'train_img', images_path, (696, 520))

    # Load validation data
    load_data(val_mta.filename, hdf5_file, 'val_labels')
    load_data(val_mta['count'], hdf5_file, 'val_counts')
    load_images(val_mta.filename, hdf5_file, 'val_img', images_path, (696, 520))

    # Load ground truth data
    load_gt(trn_mta.filename, hdf5_file, 'train_gt', images_path, (672, 512))
    load_gt(val_mta.filename, hdf5_file, 'val_gt', images_path, (672, 512))
    return hdf5_file

def quick_dataset(num_images=0, target_size=(224,224)): 
    # Get the filename metadata and set the paths of the images and ground truth files
    train_meta = pd.read_csv('C:\\Users\\Nolan\\Projects\\CSC590-CellCounting\\CellCounter\\training_set.csv')
    val_meta = pd.read_csv('C:\\Users\\Nolan\\Projects\\CSC590-CellCounting\\CellCounter\\validation_set.csv')
    gt_path = 'C:\\Users\\Nolan\\Projects\\CSC590-CellCounting\\TrainingData\\BBBC005_v1_ground_truth\\BBBC005_v1_ground_truth'
    img_path = 'C:\\Users\\Nolan\\Projects\\CSC590-CellCounting\\TrainingData\\BBBC005_v1_images\\BBBC005_v1_images'

    # Obtain arrays of the images and ground truth.  The returned arrays of images will 
    train_x = load_images(train_meta.filename[num_images:], img_path, preprocess='image', target_size=target_size)
    train_y = load_images(train_meta.filename[num_images:], gt_path, preprocess='gt', target_size=target_size)
    val_x = load_images(val_meta.filename[num_images:], img_path, preprocess='image', target_size=target_size)
    val_y = load_images(val_meta.filename[num_images:], gt_path, preprocess='gt', target_size=target_size)
    return train_x, train_y, val_x, val_y

def print_items(name, object):
    print('{0:15}'.format(name), object.shape)
    
def blob_processing(preds):
    new_preds = (preds*255).astype(dtype=np.uint8)
    return new_preds

def get_vgg_cells(radius = 2):
    # Generate the cell filenames
    tx_cell_filenames = gen_cell_filenames('cell')[:160]
    ty_dots_filenames = gen_cell_filenames('dots')[:160]
    vx_cell_filenames = gen_cell_filenames('cell')[160:]
    vy_dots_filenames = gen_cell_filenames('dots')[160:]    
    # Get the filename metadata and set the paths of the images and ground truth files
    gt_path = './Cells'
    img_path = './Cells'
    # Get some images for a dataset
    tx = process_vgg_images(img_names=tx_cell_filenames, file_path=img_path, preprocess='image', target_size=(256,256,3), mode='mask', radius=radius)
    ty  = process_vgg_images(img_names=ty_dots_filenames, file_path=gt_path, preprocess='gt', target_size=(256,256), mode='mask', radius=radius)
    vx = process_vgg_images(img_names=vx_cell_filenames, file_path=img_path, preprocess='image', target_size=(256,256,3), mode='mask', radius=radius)
    vy = process_vgg_images(img_names=vy_dots_filenames, file_path=gt_path, preprocess='gt', target_size=(256,256), mode='mask', radius=radius)
    
    #Normalize the training data by subtracting the mean pixel value for each dimension 
    avgs = per_chan_avg(tx)
    normalize(tx, avgs)
    normalize(vx, avgs)
    return tx, ty, vx, vy
    
def process_vgg_images(img_names, file_path, preprocess, target_size, mode='mask', radius=2):
    print(preprocess, preprocess == 'gt')
    if preprocess == 'image':
        images = []
        for im in img_names:
            img = Image.open(os.path.join(file_path, im))
            img = img.convert('RGB')
            img = np.asarray(img, dtype=np.float32)
            #img = image.load_img(os.path.join(file_path, im), target_size=target_size)
            #img = image.img_to_array(img)
            #img = np.expand_dims(img, axis=0)
            images.append(img)
        return np.asarray(images)
    # Ground truth processing
    elif preprocess == 'gt':
        images = []
        for im in img_names:
            img = Image.open(os.path.join(file_path, im))
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
            img = np.asarray(img, dtype=np.float32)
            if mode == 'mask':
                img = (img[:,:,0] > 0)*1
            else:
                img = img[:,:,0] / np.amax(img)
            img = np.expand_dims(img, axis=2)
            images.append(img)
        return np.asarray(images)
    
def gen_cell_filenames(cellsOrdots):
    files = []
    for i in range(3):
        if (i == 2):
            files.append('200' + str(cellsOrdots) + '.png')
            break
        for j in range(10):
            for k in range(10):
                if ((i + j + k) > 0):
                    files.append(str(i) + str(j) + str(k) + str(cellsOrdots) + '.png')
    return files 

def per_chan_avg(arr):    
    mean = [0,0,0]
    for j in range(len(mean)):
        chan_sum = 0
        for i in range(len(arr)):
            chan_sum += np.mean(arr[i][:,:,j])
        mean[j] = chan_sum / len(arr)
    return mean

def normalize(arr, avgs):
    for j in range(len(avgs)):
        for i in range(len(arr)):
            arr[i][:,:,j] -= avgs[j]

def apply_masks(imgs, masks):
    # Apply the masks to the images
    iso_imgs = (masks * imgs)
    return iso_imgs
