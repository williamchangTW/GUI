import psutil as ps
import sys
import os
import time
import numpy as np
import tensorflow # as tf
import keras
import csv
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Reshape, Conv2D, AveragePooling2D, Flatten
from keras.layers import MaxPooling2D
from keras.optimizers import adam
# import save model
from keras.models import load_model



# get file size
# get system memory infomation
def protectAvailableMemory():
    total_memory = ps.virtual_memory().total
    protect_range = int(total_memory * 0.9)
    return protect_range

# get system memory
def getSysAvailableMemory():
    return ps.virtual_memory().available

# get file size
def getFileMemoryInfo(file_name):
    return os.path.getsize(file_name)

class SizedReader:
    def __init__(self, fd, encoding='utf-8'):
        self.fd = fd
        self.size = 0
        self.encoding = encoding   # specify encoding in constructor, with utf8 as default
    def __next__(self):
        line = next(self.fd)
        self.size += len(line)
        return line.decode(self.encoding)   # returns a decoded line (a true Python 3 string)
    def __iter__(self):
        return self

def lastpointStore(file_name, data_size):
    with open(file_name, "rb") as csv_file:
        rowsize = SizedReader(csv_file)
        reader = csv.reader(rowsize)
        for row in reader:
            pos = rowsize.size
            if pos > data_size:
                with open("./checkpoints/datacheckpoint.txt", "w") as wf:
                    wf.write(str(pos))
                    break

#if os.path.exists("./checkpoints/datacheckpoint.txt") == True:
def lastpointLoad():
    with open("./checkpoints/datacheckpoint.txt", "r") as df:
        reader = df.read()
        last = int(reader)
        #last = df.seek(int(reader), 0)
        return last

def DataTestPreprocess(data_train, file_name = None):
    with open(file_name, "r") as tf:
        reader = tf.readlines()
        Lines = [line.strip().split(",") for line in reader[1:]]
        del reader
        data_test = np.array(Lines)
        del Lines
        # modified size to fit shape
        x_test = np.array(data_test[:, 1:])
    n_samples_test = x_test.shape[0]
    y_train = np.array(data_train[:, 0])
    x_train = np.array(data_train[:, 1:])
    y_train = keras.utils.to_categorical(y_train, num_classes = 10)
    #y_all_pred = np.zeros((3, n_samples_test)).astype(np.int64)
    return x_train, y_train

def dynamicLoad(file_name = None, priNumber = 15):
    # check if first read
    file_size = getFileMemoryInfo(file_name)
    with open(file_name, "r") as df:
        ava_mem = getSysAvailableMemory()
        #ava_mem = sys_mem()
        if os.path.exists("./checkpoints/datacheckpoint.txt") == True:
            lastpoint = lastpointLoad()
            if lastpoint < file_size:                
                left_data_size = file_size - lastpoint
                df.seek(lastpoint, 0)
                if left_data_size < ava_mem:
                    reader = df.readlines()
                    Lines = [line.strip().split(",") for line in reader[1:]]
                    del reader
                    data_train = np.array(Lines)
                    del Lines
                    aggsize = left_data_size + lastpoint # 目前讀取總共佔檔案大小
                    lastpointStore(file_name, aggsize)
                    del aggsize
                    # return train data and test data
                    return data_train

                elif left_data_size >= ava_mem:
                    reader = df.readlines(ava_mem) # read specific data size
                    Lines = [line.strip().split(",") for line in reader[1:]]
                    del reader
                    data_train = np.array(Lines)
                    del Lines
                    aggsize = ava_mem + lastpoint # 目前讀取總共佔檔案大小
                    lastpointStore(file_name, aggsize) # store last data size to file
                    del aggsize
                    return data_train
            else:
                raise EOFError
        elif os.path.exists("./checkpoints/datacheckpoint.txt") == False:
            if file_size < ava_mem: # 若檔案大小小於系統資源大小
                reader = df.readlines()
                Lines = [line.strip().split(",") for line in reader[1:]]
                del reader
                data_train = np.array(Lines)
                del Lines
                aggsize = file_size
                lastpointStore(file_name, aggsize)
                del aggsize
                return data_train
            
            elif file_size >= ava_mem: # 若檔案大小等於系統資源大小
                reader = df.readlines(ava_mem) # read specific data size
                Lines = [line.strip().split(",") for line in reader[1:]]
                del reader
                data_train = np.array(Lines)
                del Lines
                aggsize = ava_mem
                lastpointStore(file_name, aggsize)
                del aggsize
                return data_train

def weightSaver(model, fileName = "./checkpoints/checkpoints", platformUsed = "T"):
    # used keras as backend
    # save to json file
    # save to hd5
    if platformUsed == "K":
        model.save(fileName + ".h5")
        #print("Saved to disk!")
        # remove read checkpoint
        #os.remove("read_checkpoints.txt")
        print("Model Save!")
    
    elif platformUsed == "T":
        sess = tf.Session()
        saver = tf.train.Saver()
        saver.save(sess, fileName)
        # remove read checkpoint
        #os.remove("read_checkpoints.txt")

def weightLoader(fileName = "./checkpoints/checkpoints", platformUsed = "T"):
    # used tensorFlow as backend
    # load json file
    # load hd5
    if platformUsed == "K":
        print("load model")
        model = load_model(fileName + ".h5")
        return model
        
    elif platformUsed == "T":
        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, fileName)

def modelTrain(x_train, y_train, checkfile = None):
    if os.path.exists(checkfile + ".h5") == True:
        # test model load function
        model = weightLoader(platformUsed="K")
        model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=2, batch_size=64)
        # TODO: insert save module start
        os.remove("./checkpoints/checkpoints.h5")
        weightSaver(model, platformUsed="K")
        # TODO: insert save module end
    elif os.path.exists(checkfile + ".h5") == False or checkfile == None:
        model = Sequential()
        model.add(Reshape(target_shape=(1, 28, 28), input_shape=(784,)))
        model.add(Conv2D(kernel_size=(3, 3), filters=6, padding="same", data_format="channels_first", kernel_initializer="uniform", use_bias=False))
        model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))
        model.add(Conv2D(kernel_size=(5, 5), filters=16, padding="same", data_format="channels_first", kernel_initializer="uniform", use_bias=False))
        model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))
        model.add(Conv2D(kernel_size=(5, 5), filters=120, padding="same", data_format="channels_first", kernel_initializer="uniform", use_bias=False))
        model.add(Flatten())
        model.add(Dense(output_dim=120, activation='relu'))
        model.add(Dense(output_dim=120, activation='relu'))
        model.add(Dense(output_dim=10, activation='softmax'))

        adam = keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

        model.fit(x_train, y_train, epochs=1, batch_size=64)
        # TODO: insert save module start
        weightSaver(model, platformUsed="K")
        # TODO: insert save module end

if __name__ == "__main__":
	data = dynamicLoad("mnist_train.csv")
	x_train, y_train = DataTestPreprocess(data, "mnist_test.csv")
	modelTrain(x_train, y_train, checkfile="./checkpoints/checkpoints")
	sys.exit()






