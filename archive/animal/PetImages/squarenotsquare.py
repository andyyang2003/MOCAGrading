import pandas as pd 
import os 
from skimage.transform import resize 
from skimage.io import imread 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import svm 
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV 
from sklearn.model_selection import train_test_split 
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Define categories
Categories = ['Cube', 'NotCube']
flat_data_arr = []
target_arr = []
datadir = 'C:/Users/Andy/Desktop/moca training/cube/'

# Load images and process
for i in Categories:
    print(f'Loading... category: {i}')
    path = os.path.join(datadir, i)
    file_count = len(os.listdir(path))
    print(f'Total images in {i}: {file_count}') 
    
    for idx, img in enumerate(os.listdir(path)):
        try:
            img_path = os.path.join(path, img)
            img_array = imread(img_path)
            img_resized = resize(img_array, (84, 84, 3))
            img_normalized = img_resized / 255 # normalize the image data to 0-1 
            flat_data_arr.append(img_normalized.flatten())
            target_arr.append(Categories.index(i))
        except Exception as e:
            print(f'Failed to process image {img}: {e}')
    
    print(f'Loaded category {i} successfully')

# Convert to numpy arrays and create DataFrame
flat_data = np.array(flat_data_arr)
target = np.array(target_arr)
df = pd.DataFrame(flat_data)
df['Target'] = target
print(f'DataFrame shape: {df.shape}')

# Save DataFrame to CSV
csv_filename = 'cube_dataset.csv'
df.to_csv(csv_filename, index=False)
print(f'Dataset saved as {csv_filename}')

# Split data into training and testing sets
x = df.iloc[:, :-1]  # input data
y = df.iloc[:, -1]   # output data
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size=0.2,  # use a larger test size to speed up
                                                    random_state=77,
                                                    stratify=y)
print('Data split into training and testing sets.')

# Defining the parameters grid for GridSearchCV 
param_grid={'C':[0.1,1,10,100], 
			'gamma':[0.0001,0.001,0.1,1], 
			'kernel':['linear', 'rbf', 'poly']} 

# Creating a support vector classifier 
svc=svm.SVC(probability=True) 

# Creating a model using GridSearchCV with the parameters grid 
model=GridSearchCV(svc,param_grid)

print('Starting model training...')
model.fit(x_train, y_train)
print('Model training completed.')

# Predict and evaluate the model (optional)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_pred, y_test)
print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(classification_report(y_test, y_pred, target_names=Categories))

# Save the model
model_filename = 'cube_classifier_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)
print(f'Model saved as {model_filename}')




