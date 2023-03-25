import h5py 
import numpy as np
from keras.layers import Conv3D,BatchNormalization,Flatten,Dense,Dropout,MaxPooling3D  
from keras.models import Model,Sequential
from keras.initializers import Constant
from tensorflow.keras.optimizers import Adam
from keras.callbacks import Callback,EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

data_file =r"C:\Users\babaa\Downloads\mnist 3d\full_dataset_vectors.h5"

with h5py.File(data_file,'r') as dataset:
    x_train = dataset["X_train"][:]
    x_test = dataset["X_test"][:]
    y_train = dataset["y_train"][:]
    y_test = dataset["y_test"][:]
   
# print('x_train shape:', x_train.shape)
# print('y_train shape:', y_train.shape)
# print(y_train.shape)



xtrain = np.array(x_train)
xtest = np.array(x_test)

# print(xtrain.shape)
# print(xtest.shape)

xtrain = xtrain.reshape(xtrain.shape[0],16,16,16,1)
xtest = xtest.reshape(xtest.shape[0],16,16,16,1)

# print(xtest.shape)
# print(y_train.shape)
ytrain = to_categorical(y_train,10)
ytest = to_categorical(y_test,10)
# print(ytrain.shape)



model = Sequential()
model.add(Conv3D(32,(3,3,3),activation='relu',input_shape=(16,16,16,1),bias_initializer=Constant(0.01)))
model.add(Conv3D(32,(3,3,3),activation='relu',bias_initializer=Constant(0.01)))
model.add(MaxPooling3D((2,2,2)))
model.add(Conv3D(64,(3,3,3),activation='relu'))
model.add(Conv3D(64,(2,2,2),activation='relu'))
model.add(MaxPooling3D((2,2,2)))
model.add(Dropout(0.6))
model.add(Flatten())
model.add(Dense(256,'relu'))
model.add(Dropout(0.7))
model.add(Dense(128,'relu'))
model.add(Dropout(0.5))
model.add(Dense(10,'softmax'))
# print(model.summary())
model.compile(Adam(lr=0.001),'categorical_crossentropy',['acc'])
training_model = model.fit(xtrain,ytrain,epochs=200,batch_size=32,verbose=1,validation_data=(xtest,ytest),callbacks=[EarlyStopping(patience=15)])

model_save = training_model.model.save("3D CNN_100.h5")



