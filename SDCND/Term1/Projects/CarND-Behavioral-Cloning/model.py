
# coding: utf-8

# In[44]:

import numpy as np
import preprocess
from preprocess import *

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Convolution2D
from keras.layers.core import Lambda
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2
from keras.layers.advanced_activations import ELU

from keras.optimizers import Adam 

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import csv, cv2
import pandas as pd
import random
import os


# In[2]:

imgdir_path = './data/'
csv_filepath = 'driving_log.csv'
img_dirs = ['udacity', 'curve_1', 'curve_2_1', 'curve_2_2',
				'curve_3_1', 'curve_3_2', 'dust_curve_1',
				'dust_curve_2', 'river_1', 'river_2']


# In[4]:

for i in img_dirs:
	csv_file = imgdir_path + i + '/' + i + '_' + csv_filepath
	if i == 'udacity':
		data = pd.read_csv(csv_file)
		data.columns = ["center", "left", "right", "steering", "throttle", "brake", "speed"]
		continue
	
	tmp_data = pd.read_csv(csv_file)
	tmp_data.columns = ["center", "left", "right", "steering", "throttle", "brake", "speed"]
	tmp_data = tmp_data[tmp_data.steering != 0]
	data = data.append(tmp_data)
	
data = data.drop(["throttle", "brake", "speed"], axis=1)
print("Total training data: ", data.shape[0])

# In[18]:

data = shuffle(data)
data = data.reset_index(drop=True)
train_data, valid_data = train_test_split(data, test_size=0.2)


# In[39]:

nvidia_img_h = 66
nvidia_img_w = 220
nvidia_c = 3
camera_pos = ['left','center','right']
steer_offset = {'left':0.25, 'center':0., 'right':-0.25}
input_shape = (66,200,3)

def train_data_generator(df, batch_size = 128):
	batch_images = np.zeros((batch_size, *input_shape))
	batch_angles = np.zeros(batch_size)
	df = shuffle(df)
	df = df.reset_index(drop=True)
	
	while True:
		for i in range(batch_size):
			idx = np.random.randint(len(df))

			row = df.iloc[[idx]].reset_index()
			
			c_pos = random.choice(camera_pos)

			f = imgdir_path + row[c_pos][0].split('/')[0].strip('_IMG') + '/' + row[c_pos][0]
			if os.path.isfile(f) == False:
				print("file doesnt exist: ", f)
				exit()

			img = cv2.imread(f)
			angle = row['steering'][0]
			
			if (random.choice([0,1])==0) :
				img = np.fliplr(img)
				angle = -angle

			angle = angle + steer_offset[c_pos]

			img = preprocess_img(img)
			

			batch_images[i] = img
			batch_angles[i] = angle

		yield (batch_images, batch_angles)
	
def valid_data_generator(df, batch_size = 128):
	batch_images = np.zeros((batch_size, *input_shape))
	batch_angles = np.zeros(batch_size)
	df = shuffle(df)
	df = df.reset_index(drop=True)
	
	while True:
		for i in range(batch_size):
			idx = np.random.randint(len(df))

			row = df.iloc[[idx]].reset_index()
			
			c_pos = random.choice(camera_pos)

			f = imgdir_path + row[0][0].split('/')[0].strip('_IMG') + '/' + row[0][0]
			if os.path.isfile(f) == False:
				print("file doesnt exist: ", f)
				exit()

			img = cv2.imread(f)
			angle = row['steering'][0]
			
			img = preprocess_img(img)

			batch_images[i] = img
			batch_angles[i] = angle

		yield (batch_images, batch_angles)

# In[40]:

# compile and train the model using the generator function
train_generator = train_data_generator(train_data, batch_size=128)
valid_generator = valid_data_generator(valid_data, batch_size=128)

#validation_generator = validation_data_generator(valid_data)

# In[33]:

# Based on NVIDIA model

def get_model_nvidia():
	model = Sequential()

	model.add(Lambda(lambda x: x/127.5 - 1.0,input_shape=(66,200,3)))

	model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
	model.add(ELU())
	model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
	model.add(ELU())
	model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
	model.add(ELU())

	model.add(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001)))
	model.add(ELU())
	model.add(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001)))
	model.add(ELU())

	model.add(Flatten())

	model.add(Dense(100, W_regularizer=l2(0.001)))
	model.add(ELU())
	model.add(Dense(50, W_regularizer=l2(0.001)))
	model.add(ELU())
	model.add(Dense(10, W_regularizer=l2(0.001)))
	model.add(ELU())

	# Add a fully connected output layer
	model.add(Dense(1))

	model.compile(optimizer=Adam(lr=1e-4), loss='mse')
	
	model.summary()
	return model

# In[45]:

batch_size = 128
nb_epoch = 5

checkpointer = ModelCheckpoint(filepath="./tmp/nvidia.{epoch:02d}.hdf5",
								verbose=1, save_best_only=False)

model = get_model_nvidia()

train_samples_per_epoch = data.shape[0] - (data.shape[0] % batch_size)

history = model.fit_generator(train_generator,
							  samples_per_epoch=train_samples_per_epoch,
							  nb_epoch=nb_epoch,
							  validation_data=valid_generator,
							  nb_val_samples= valid_data.shape[0],
							  verbose=1,
							  callbacks=[checkpointer])
# In[ ]:

### print the keys contained in the history object
print(history.history.keys())


# In[ ]:
'''
### plot the training and validation loss for each epoch
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
'''

# In[ ]:

model_json = model.to_json()
with open("model.json", "w") as json_file:
	json_file.write(model_json)

model.save("model.h5")
print("Saved model to disk")
