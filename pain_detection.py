from future
import division, absolute_import
import numpy as np
import tflearn
from tflearn.layers.core 
import input_data, dropout, fully_connected, flatten22SoftwareRequirements
from tflearn.layers.conv
import conv_2d, max_pool_2d, avg_pool_2d from tflearn.layers.merge_ops import merge
from tflearn.layers.normalization import local_response_normalization 
from tflearn.layers.estimator import regression
from os.path import isfile, join 
import sys
import tensorflow as tf
import os
# prevents appearance of tensorflow warnings 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
tf.logging.set_verbosity(tf.logging.ERROR)
class EMR:
def init(self):
self.target_classes = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
def build_network(self): """Build the convnet. Input is 48x483072 nodes in fully connected layer """
self.network = input_data(shape = [None, 48, 48, 1]) 
print("Input data ",self.network.shape[1:])
self.network = conv_2d(self.network, 64, 5, activation = 'relu') 
print("Conv1 ",self.network.shape[1:])
self.network = max_pool_2d(self.network, 3, strides = 2) 
print("Maxpool1 ",self.network.shape[1:])
self.network = conv_2d(self.network, 64, 5, activation = 'relu') 
print("Conv2 ",self.network.shape[1:])
self.network = max_pool_2d(self.network, 3, strides = 2) 
print("Maxpool2 ",self.network.shape[1:])
self.network = conv_2d(self.network, 128, 4, activation = 'relu') 
print("Conv3 ",self.network.shape[1:])
self.network = dropout(self.network, 0.3) 
print("Dropout ",self.network.shape[1:])
self.network = fully_connected(self.network, 3072, activation = 'relu') 
print("Fully connected",self.network.shape[1:])
self.network = fully_connected(self.network, len(self.target_classes), activation = 'softmax')
print("Output ",self.network.shape[1:]) 
print("\n")
# Generates a TrainOp which contains the information about optimization process -
optimizer, loss function, etc
self.network = regression(self.network,optimizer = 'momentum',metric = 
'accuracy',loss = 'categorical_crossentropy')
# Creates a model instance.23
SoftwareRequirements Specificationfor<Project> Page 23
self.model = tflearn.DNN(self.network,checkpoint_path = 
'model_1_atul',max_checkpoints = 1,tensorboard_verbose = 2)
# Loads the model weights from the checkpoint 
self.load_model()
def predict(self, image):
"""Image is resized to 48x48, and predictions are returned."""
if image is None:
return None
image = image.reshape([-1, 48, 48, 1])
return self.model.predict(image)
def load_model(self):
"""Loads pre-trained model. """
if isfile("model_1_atul.tflearn.meta"): 
self.model.load("model_1_atul.tflearn") 
else:
print("---> Couldn't find model")
if name == "main":
print("\n------------Pain Detection Program ----------------------\n")
network = EMR()
if sys.argv[1] == 'singleface': 
import singleface
if sys.argv[1] == 'multiface': 
import multiface24
SoftwareRequirements Specificationfor<Project> Page 24
Part 2:
\import cv2 
import sys
import numpy as np from 
model import EMR
# prevents opencl usage and unnecessary logging messages 
cv2.ocl.setUseOpenCL(False)
EMOTIONS = ['angry', 'disgusted', 'fearful', 'smile', 'sad', 'surprised', 'neutral']
def format_image(image): """Function to format frame """
if len(image.shape) > 2 and image.shape[2] == 3: # determine whether the image is color
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) else:
# Image read from buffer
image = cv2.imdecode(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)
cascade_classifier = 
cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces = cascade_classifier.detectMultiScale(image,scaleFactor = 1.3
,minNeighbors = 5)
if not len(faces) > 0: 
return None
# initialize the first face as having maximum area, then find the one with max_area
max_area_face = faces[0] for 
face in faces:
if face[2] * face[3] > max_area_face[2] * max_area_face[3]: 
max_area_face = face
face = max_area_face
# extract ROI of face
image = image[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]
try:25
SoftwareRequirements Specificationfor<Project> Page 25
# resize the image so that it can be passed to the neural network
image = cv2.resize(image, (48,48), interpolation = cv2.INTER_CUBIC) /
255.
except Exception:
print(" ------ >Problem during resize")
return None
return image
# Initialize object of EMR class 
network = EMR() 
network.build_network()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
feelings_faces = []
# append the list with the emoji images
for index, emotion in enumerate(EMOTIONS): 
feelings_faces.append(cv2.imread('./emojis/' + emotion + '.png', -1))
while True:
# Again find haar cascade to draw bounding box around face ret, frame 
= cap.read()
if not ret: 
break
facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') gray = 
cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
# compute softmax probabilities
result = network.predict(format_image(frame)) 
if result is not None:
expsum=result[0][0]+result[0][1]+result[0][2]+result[0][4]+result[0][5]+result[ 0][6]
# write the different emotions and have a bar to indicate probabilities for each class
if result[0][3]>0.7 and expsum<0.25: pain = 0
else:
pain26=3*result[0][4]+3*result[0][0]+1*result[0][1]+4*result[0][2]+1.5*result[0][3]+ 0.5*result[0][5]-0.05*result[0][6]
for index, emotion in enumerate(EMOTIONS):
cv2.putText(frame, emotion, (10, index * 20 + 20),
cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1);
cv2.rectangle(frame, (130, index * 20 + 10), (130 +
int(result[0][index] * 100), (index + 1) * 20 + 4), (255, 0, 0), -1)
# find the emotion with maximum probability and display it 
#maxindex = np.argmax(result[0])
#font = cv2.FONT_HERSHEY_SIMPLEX
#cv2.putText(frame,EMOTIONS[maxindex],(10,200), font, 2,(255,255,255),2,cv2.LINE_AA)
#face_image = feelings_faces[maxindex]
#painlevel of the person along with the bar 
if(pain<1.35):
cv2.putText(frame,"Pain level:"+str(pain),(10,350), font, 
1,(255,255,255),2,cv2.LINE_AA)
elif(pain>=1.35 and pain<2.1):
cv2.putText(frame,"Pain level:"+str(pain),(10,350), font, 
1,(0,255,255),2,cv2.LINE_AA)
else:
cv2.putText(frame,"Pain level:"+str(pain),(10,350), font, 
1,(0,0,255),2,cv2.LINE_AA)
cv2.rectangle(frame, (115, 400), (115+ int(pain*90), 380), (255, 0, 0),-1)
#for c in range(0, 3):
# The shape of face_image is (x,y,4). The fourth channel is 0 or 1. In most cases it is 0, so, we assign the roi to the emoji.
# You could also do: frame[200:320,10:130,c] = frame[200:320, 10:130, c] 
#(1.0 - face_image[:, :, 3] / 255.0)
#frame[200:320, 10:130, c] = face_image[:,:,c]*(face_image[:, :, 3]
#/ 255.0) + frame[200:320, 10:130, c] * (1.0 - face_image[:, :, 3] / 255.0)
if len(faces) > 0:
# draw box around face with maximum area 27
SoftwareRequirements Specificationfor<Project> Page 27
max_area_face = faces[0]
for face in faces:
if face[2] * face[3] > max_area_face[2] * max_area_face[3]: 
max_area_face = face
face = max_area_face (x,y,w,h) = max_area_face
frame = cv2.rectangle(frame,(x,y-50),(x+w,y+h+10),(255,0,0),2)
cv2.imshow('Video', cv2.resize(frame,None,fx=2,fy=2,interpolation = 
cv2.INTER_CUBIC))
if cv2.waitKey(1) & 0xFF == ord('q'): 
    break
cap.release() 
cv2.destroyAllWindows()
#pain = 6*result[0][4]+4*result[0][0]+2*result[0][1]+result[0][2]-4*result[0][3]+2*result[0][5]