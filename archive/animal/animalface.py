import matplotlib.pyplot as plt
import numpy as np
import os
import pprint
import joblib
from skimage.io import imread
from skimage.transform import resize
pp = pprint.PrettyPrinter(indent=4)
def resize_all(src, pklname, include, width=150, height=None):
    """
    Resize all images in a directory and save them to a pkl file
    src: directory of images
    pklname: name of pkl file
    include: tuple of allowed extensions
    width: new width
    height: new height
    """
    height = height if height is not None else width
    data = dict()
    data['description'] = 'resized ({0}x{1}) animal images in rgb'.format(int(width), int(height))
    data['label'] = []
    data['filename'] = []
    data['data'] = []
    pklname = f"{pklname}_{width}x{height}px.pkl"
    for subdir in os.listdir(src):
        if subdir in include:
            current_path = os.path.join(src, subdir)
            print(subdir)

            for file in os.listdir(current_path):
                if file[-3:] in {'jpg', 'png'}:
                    img = imread(os.path.join(current_path, file))
                    img = resize(img, (width, height))
                    data['label'].append(subdir[:-4])
                    data['filename'].append(file)
                    data['data'].append(img)
                
        joblib.dump(data, pklname)      
        
data_path = 'C:/Users/Andy/desktop/moca training/AnimalFace/Image'
print(os.listdir(data_path))
base_name = 'animal_faces'
width = 80
include = {'ChickenHead', 'BearHead', 'ElephantHead', 'EagleHead', 'DeerHead', 'MonkeyHead', 'PandaHead'}
resize_all(data_path, base_name, include, width)
#resize_all("C:/Users/Andy/Desktop/moca training/AnimalFace/Image", 'animal_faces', ('cat', 'dog', 'panda'), 50)
from collections import Counter
data = joblib.load(f'{base_name}_{width}x{width}px.pkl')
print('number of samples: ', len(data['data']))
print('keys: ', list(data.keys()))
print('description: ', data['description'])
print('image shape: ', data['data'][0].shape)
print('labels:', np.unique(data['label']))

print(Counter(data['label']))

labels = np.unique(data['label'])
fig, axes = plt.subplots(1, len(labels))
fig.set_size_inches(15, 4)
fig.tight_layout()

for ax, label in zip(axes, labels):
    idx = data['label'].index(label)
    ax.imshow(data['data'][idx])
    ax.axis('off')
    ax.set_title(label)