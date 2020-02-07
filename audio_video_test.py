import numpy as np
import tensorflow as tf
from sklearn import svm
import sys, math, os
import h5py
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
from sklearn.model_selection import GridSearchCV
from keras.layers import Dense, LSTM, TimeDistributed, Dropout, BatchNormalization, Activation, Add, Input
from keras.layers import Bidirectional, concatenate, GRU, LeakyReLU, PReLU 
from keras.initializers import TruncatedNormal, VarianceScaling
from keras.initializers import Constant
from keras import regularizers
from keras.models import Sequential, Model
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from keras.callbacks import History 
history = History()
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.models import model_from_json
from keras.backend import manual_variable_initialization 
from keras.optimizers import SGD, RMSprop
from numpy import array
from numpy import argmax
from keras.utils import to_categorical
import pandas as pd



def splitting(x,window_size=15,overlap_size=5,pad=True):
	 val = []
	 j = 0
	 if(len(x)<window_size and pad):
		 to_add = np.zeros((window_size-len(x)+1,1))
		 x=np.concatenate((x,to_add), axis=0)
	 while(j<len(x)-window_size):
		 val.append(np.array(x[j:j+window_size]))
		 j+=window_size-overlap_size+1
	 return (val)
 

train_data_audio = np.load('/home/ubuntu/Desktop/Sowmya/AffWild2/audio_training_data.npy', allow_pickle=True)
val_data_audio = np.load('/home/ubuntu/Desktop/Sowmya/AffWild2/validation_data.npy',  allow_pickle=True)
test_data_audio= np.load('/home/ubuntu/Desktop/Sowmya/AffWild2/test_data_audio_old.npy',  allow_pickle=True)


audio_data=np.vstack([train_data_audio,val_data_audio])
audio_data = np.delete(audio_data, 2, 1) 
full_data=np.vstack([audio_data,test_data_audio])

print("train",train_data_audio.shape)
print("val",val_data_audio.shape)
print("full",full_data.shape)


x_train_audio_names=[]
for i in range(len(full_data)):
	temp = np.array(full_data[i][0])
	x_train_audio_names.append(temp)
print("length",len(x_train_audio_names))

train_full_data=[]
for i in range(len(full_data)):
	temp = np.array(full_data[i][1])
	train_full_data.append(temp)

train_data_csv = np.load('/home/ubuntu/Desktop/Sowmya/AffWild2/feat_new/train_feat_face.npy', allow_pickle=True)
val_data_csv = np.load('/home/ubuntu/Desktop/Sowmya/AffWild2/feat_new/val_feat_face.npy',  allow_pickle=True)
test_data_csv = np.load('/home/ubuntu/Desktop/Sowmya/AffWild2/feat_new/test_feat_face_old.npy',  allow_pickle=True)
txt_dir="/home/ubuntu/Desktop/Sowmya/AffWild2/annotations/EXPR_Set/Training_Set/"
val_dir="/home/ubuntu/Desktop/Sowmya/AffWild2/annotations/EXPR_Set/Validation_Set/"
test_dir="/home/ubuntu/Desktop/Sowmya/AffWild2/Expression/expression_test_set.txt"

print('Video train', np.array(train_data_csv).shape)
print('Video val', np.array(val_data_csv).shape)
train_video=np.reshape(train_data_csv, np.array(train_data_csv).shape + (1,))
val_video = np.reshape(val_data_csv, np.array(val_data_csv).shape + (1,))
video_data = np.vstack([train_video, val_video])
temp = np.delete(test_data_csv, 0, 1)
# temp = np.reshape(temp, (np.array(temp).shape[0] + (1, ))
video_data_full = np.vstack([video_data, temp])
flat = []
for i in video_data_full:
	for j in i:
		flat.append(j)
print(np.array(flat[1]).shape)

def test(input_dir,x_train_audio_names,train_full_data,k):
	val_arou_Train=[]
	with open(input_dir) as f:
		files=f.readlines()
		lines = [l.strip() for l in files if l.strip()]
	print("length of lines",lines)
	for i in range(len(x_train_audio_names)):
			for line in lines:
				if line == x_train_audio_names[i]:
					#print('File matched',x_train_audio_names[i])
					x = train_full_data[i]
					#print(len(x))    
					val_arou_Train.append([x,x_train_audio_names[i]])
	
	return val_arou_Train


def av_model(filename,audio_data,face_data):

	x_test_audio=np.array(audio_data)
	x_test_face=np.array(face_data)
	print("shapes")
	print(x_test_audio.shape,x_test_face.shape)
	model=load_model("/home/ubuntu/Desktop/Sowmya/AffWild2/Expression/gru_model.h5")
	predicted_output = model.predict([x_test_audio,x_test_face],batch_size=100)
	#pred_out = predicted_output.argmax(axis=-1)
	print(predicted_output.shape)
	text_file_gen(predicted_output,filename)

def text_file_gen(predicted_output,filename):

	data = predicted_output

	print(filename)
	filename="/home/ubuntu/Desktop/Sowmya/AffWild2/expression_text_files/"+str(filename)+".txt"
	print(filename)

	# Write the array to disk
	with open(filename, 'w') as outfile:

		for i in range(len(data)):
			
		
			if i==0:
				pre_data=data[i]
				#print("pd",pre_data)
				pre_data=pre_data[:-5,:]
				d_slice=desplitting(data[i],data[i+1])
				data_slice=np.vstack([pre_data,d_slice])
			elif i==len(data)-1:
				pre_data=data[i]
				pre_data=pre_data[5:,:]
				print(pre_data.shape)
				data_slice=pre_data
			else:
				pre_data=data[i]
				pre_data=pre_data[:-5,:]
				pre_data=pre_data[5:,:]
				d_slice=desplitting(data[i],data[i+1])
				data_slice=np.vstack([pre_data,d_slice])
			data_slice = data_slice.argmax(axis=-1)
			data_slice[data_slice == 7] = -1
			#print("data",data_slice)
			np.savetxt(outfile, data_slice, fmt='%-7.0f')
	
def desplitting(data1,data2):
	append1=data1[10:, :]
	append2=data2[:-10, :]
	avg_res=np.mean( np.array([ append1, append2 ]), axis=0 )
	return(avg_res)

test_audio=test(test_dir,x_train_audio_names,train_full_data,168)
test_video=test(test_dir,x_train_audio_names,flat,714)
for i in range(len(test_audio)):

				filename=np.array(test_audio[i][1])
				#print("filename",filename)
				audio_data=np.array(test_audio[i][0])
				face_data=np.array(test_video[i][0])	
				av_model(filename,audio_data,face_data)



