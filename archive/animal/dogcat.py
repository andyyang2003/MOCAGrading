import pandas as pd 
import os 
from skimage.transform import resize 
from skimage.io import imread 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import svm 
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
import pickle

Categories = ['CatHead', 'DogHead']
flat_data_arr = []
target_arr = []
datadir = 'C:/Users/Andy/Desktop/moca training/animal/PetImages/'

for i in Categories:
    print(f'Loading... category: {i}')
    path = os.path.join(datadir, i)
    file_count = len(os.listdir(path))
    print(f'Total images in {i}: {file_count}')
    
    for idx, img in enumerate(os.listdir(path)):
        try:
            img_path = os.path.join(path, img)
            img_array = imread(img_path)
            img_resized = resize(img_array, (150, 150, 3))
            flat_data_arr.append(img_resized.flatten())
            target_arr.append(Categories.index(i))
        except Exception as e:
            print(f'Failed to process image {img}: {e}')
    
    print(f'Loaded category {i} successfully')

flat_data = np.array(flat_data_arr)
target = np.array(target_arr)
df = pd.DataFrame(flat_data)
df['Target'] = target
print(f'DataFrame shape: {df.shape}') #created dataframe

csv_filename = 'catdog_dataset.csv'
df.to_csv(csv_filename, index=False)
print(f'Dataset saved as {csv_filename}')

x = df.iloc[:, :-1] # input data
y = df.iloc[:, -1] # output data
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size=0.1,  # use a larger test size to speed up
                                                    random_state=77,
                                                    stratify=y)
# Split data into training and testing

# Build SVM with Randomized Search CV 
param_grid = {'C': [0.1, 1, 10],
              'gamma': [1, 0.1, 0.01, 0.001],
              'kernel': ['rbf', 'linear']}

svc = svm.SVC(probability=True)
#model = RandomizedSearchCV(svc, param_grid, n_iter=5, verbose=3)  # reduce n_iter for faster search
model = GridSearchCV(svc, param_grid, refit=True, verbose=3)
print('Starting model training...')
model.fit(x_train, y_train)
print('Model training completed.')

best_model = model.best_estimator_
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_pred, y_test)
print(f'Accuracy: {accuracy}')
model_filename = 'best_model_classifier.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(best_model, f)
print(f"pickle saved as {model_filename}")
