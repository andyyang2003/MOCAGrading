import numpy as np
from skimage.transform import resize 
from skimage.io import imread 
import matplotlib.pyplot as plt
import pickle
model_name = 'cube_classifier_model.pkl'
with open(model_name, 'rb') as f:
    model = pickle.load(f)
Categories = ['Cube', 'NotCube']

#path="C://Users//Andy//Desktop//moca training//cube//NotCube//3022c.png"
#path="C://Users//Andy//Desktop//moca training//cube4.jpg"
#path="C://Users//Andy//Desktop//moca training//rectangle.png"
#path="C://Users//Andy//Desktop//moca training//cube.png"
path="C://Users//Andy//Desktop//moca training//animal//PetImages//DogHead//dog000001c.jpg"
#path="C://Users//Andy//Desktop//hexprism.png"
#path="C://Users//Andy//Desktop//polyhedron.png"
img=imread(path) 
plt.imshow(img) 
plt.show(block=True)
img_resize=resize(img,(84,84,3)) 
l=[img_resize.flatten()] 
probability=model.predict_proba(l) 
for ind,val in enumerate(Categories): 
    print(f'{val} = {probability[0][ind]*100}%  ') 
print("The predicted image is : "+Categories[model.predict(l)[0]])

