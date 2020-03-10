import os
import numpy as np
import matplotlib as plt
import cv2
import glob
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPooling2D,Dropout,Activation,Input,Add,BatchNormalization
from tensorflow.keras.layers import ZeroPadding2D,AveragePooling2D
from tensorflow.keras.initializers import glorot_uniform


filepath1="/home/vinay/Documents/Image Classification/data/train/airplane/*.*"
filepath2="/home/vinay/Documents/Image Classification/data/train/airport/*.*"
filepath3="/home/vinay/Documents/Image Classification/data/train/basketball_diamond/*.*"
filepath4="/home/vinay/Documents/Image Classification/data/train/basketball_court/*.*"
filepath5="/home/vinay/Documents/Image Classification/data/train/beach/*.*"
filepath6="/home/vinay/Documents/Image Classification/data/train/bridge/*.*"
filepath7="/home/vinay/Documents/Image Classification/data/train/chaparral/*.*"
filepath8="/home/vinay/Documents/Image Classification/data/train/church/*.*"
filepath9="/home/vinay/Documents/Image Classification/data/train/circular_farmland/*.*"
filepath10="/home/vinay/Documents/Image Classification/data/train/commercial_area/*.*"
filepath11="/home/vinay/Documents/Image Classification/data/train/dense_residential/*.*"
filepath12="/home/vinay/Documents/Image Classification/data/train/desert/*.*"
filepath13="/home/vinay/Documents/Image Classification/data/train/golf_course/*.*"
filepath14="/home/vinay/Documents/Image Classification/data/train/ground_track_field/*.*"
filepath15="/home/vinay/Documents/Image Classification/data/train/harbor/*.*"
filepath16="/home/vinay/Documents/Image Classification/data/train/industrial_area/*.*"
filepath17="/home/vinay/Documents/Image Classification/data/train/intersection/*.*"
filepath18="/home/vinay/Documents/Image Classification/data/train/forest/*.*"
filepath19="/home/vinay/Documents/Image Classification/data/train/freeway/*.*"
filepath20="/home/vinay/Documents/Image Classification/data/train/island/*.*"
filepath21="/home/vinay/Documents/Image Classification/data/train/lake/*.*"
filepath22="/home/vinay/Documents/Image Classification/data/train/meadow/*.*"
filepath23="/home/vinay/Documents/Image Classification/data/train/medium_residential/*.*"
filepath24="/home/vinay/Documents/Image Classification/data/train/mobile_home_park/*.*"
filepath25="/home/vinay/Documents/Image Classification/data/train/mountain/*.*"
filepath26="/home/vinay/Documents/Image Classification/data/train/overpass/*.*"
filepath27="/home/vinay/Documents/Image Classification/data/train/parking_lot/*.*"
filepath28="/home/vinay/Documents/Image Classification/data/train/railway/*.*"
filepath29="/home/vinay/Documents/Image Classification/data/train/rectangular_farmland/*.*"
filepath30="/home/vinay/Documents/Image Classification/data/train/roundabout/*.*"
filepath31="/home/vinay/Documents/Image Classification/data/train/runway/*.*"





filepath01="/home/vinay/Documents/Image Classification/data/test/airplane/*.*"
filepath02="/home/vinay/Documents/Image Classification/data/test/airport/*.*"
filepath03="/home/vinay/Documents/Image Classification/data/test/basketball_diamond/*.*"
filepath04="/home/vinay/Documents/Image Classification/data/test/basketball_court/*.*"
filepath05="/home/vinay/Documents/Image Classification/data/test/beach/*.*"
filepath06="/home/vinay/Documents/Image Classification/data/test/bridge/*.*"
filepath07="/home/vinay/Documents/Image Classification/data/test/chaparral/*.*"
filepath08="/home/vinay/Documents/Image Classification/data/test/church/*.*"
filepath09="/home/vinay/Documents/Image Classification/data/test/circular_farmland/*.*"
filepath010="/home/vinay/Documents/Image Classification/data/test/commercial_area/*.*"
filepath011="/home/vinay/Documents/Image Classification/data/test/dense_residential/*.*"
filepath012="/home/vinay/Documents/Image Classification/data/test/desert/*.*"
filepath013="/home/vinay/Documents/Image Classification/data/test/golf_course/*.*"
filepath014="/home/vinay/Documents/Image Classification/data/test/ground_track_field/*.*"
filepath015="/home/vinay/Documents/Image Classification/data/test/harbor/*.*"
filepath016="/home/vinay/Documents/Image Classification/data/test/industrial_area/*.*"
filepath017="/home/vinay/Documents/Image Classification/data/test/intersection/*.*"
filepath018="/home/vinay/Documents/Image Classification/data/test/forest/*.*"
filepath019="/home/vinay/Documents/Image Classification/data/test/freeway/*.*"
filepath020="/home/vinay/Documents/Image Classification/data/test/island/*.*"
filepath021="/home/vinay/Documents/Image Classification/data/test/lake/*.*"
filepath022="/home/vinay/Documents/Image Classification/data/test/meadow/*.*"
filepath023="/home/vinay/Documents/Image Classification/data/test/medium_residential/*.*"
filepath024="/home/vinay/Documents/Image Classification/data/test/mobile_home_park/*.*"
filepath025="/home/vinay/Documents/Image Classification/data/test/mountain/*.*"
filepath026="/home/vinay/Documents/Image Classification/data/test/overpass/*.*"
filepath027="/home/vinay/Documents/Image Classification/data/test/parking_lot/*.*"
filepath028="/home/vinay/Documents/Image Classification/data/test/railway/*.*"
filepath029="/home/vinay/Documents/Image Classification/data/test/rectangular_farmland/*.*"
filepath030="/home/vinay/Documents/Image Classification/data/test/roundabout/*.*"
filepath031="/home/vinay/Documents/Image Classification/data/test/runway/*.*"




def read_train_file():
    train_data=[]
    x_train=[]
    y_train=[]
    i=0
    for file in glob.glob(filepath1):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(0)
        train_data.append([a,"airplane"])
    for file in glob.glob(filepath2):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(1)
        train_data.append([a,"airport"])
    for file in glob.glob(filepath3):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(2)
        train_data.append([a,"basketball_diamond"])
    for file in glob.glob(filepath4):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(3)
        train_data.append([a,"basketball_court"])
    for file in glob.glob(filepath5):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(4)
        train_data.append([a,"beach"])
    for file in glob.glob(filepath6):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(5)
        train_data.append([a,"bridge"])
    for file in glob.glob(filepath7):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(6)
        train_data.append([a,"chaparral"])
    for file in glob.glob(filepath8):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(7)
        train_data.append([a,"church"])
    for file in glob.glob(filepath9):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(8)
        train_data.append([a,"circular_farmland"])
    for file in glob.glob(filepath10):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(9)
        train_data.append([a,"commercial_area"])
    for file in glob.glob(filepath11):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(10)
        train_data.append([a,"dense_residential"])
    for file in glob.glob(filepath12):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(11)
        train_data.append([a,"desert"])
    for file in glob.glob(filepath13):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(12)
        train_data.append([a,"golf_course"])
    for file in glob.glob(filepath14):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(13)
        train_data.append([a,"ground_track_field"])
    for file in glob.glob(filepath15):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(14)
        train_data.append([a,"harbor"])
    for file in glob.glob(filepath16):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(15)
        train_data.append([a,"industrial_area"])
    for file in glob.glob(filepath17):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(16)
        train_data.append([a,"intersection"])
    for file in glob.glob(filepath18):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(17)
        train_data.append([a,"forest"])
    for file in glob.glob(filepath19):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(18)
        train_data.append([a,"freeway"])
    for file in glob.glob(filepath20):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(19)
        train_data.append([a,"island"])
    for file in glob.glob(filepath21):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(20)
        train_data.append([a,"lake"])
    for file in glob.glob(filepath22):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(21)
        train_data.append([a,"meadow"])
    for file in glob.glob(filepath23):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(22)
        train_data.append([a,"medium_residential"])
    for file in glob.glob(filepath24):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(23)
        train_data.append([a,"mobile_home_park"])
    for file in glob.glob(filepath25):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(24)
        train_data.append([a,"mountain"])
    for file in glob.glob(filepath26):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(25)
        train_data.append([a,"overpass"])
    for file in glob.glob(filepath27):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(26)
        train_data.append([a,"parking_lot"])
    for file in glob.glob(filepath28):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(27)
        train_data.append([a,"railway"])
    for file in glob.glob(filepath29):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(28)
        train_data.append([a,"rectangular_farmland"])
    for file in glob.glob(filepath30):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(29)
        train_data.append([a,"round_about"])
    for file in glob.glob(filepath31):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(30)
        train_data.append([a,"runway"])
    
    return np.asarray(x_train),np.asarray(y_train)
    # return x_train,y_train
    
def read_test_file():
    train_data=[]
    x_train=[]
    y_train=[]
    i=0
    for file in glob.glob(filepath01):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(0)
        train_data.append([a,"airplane"])
    for file in glob.glob(filepath02):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(1)
        train_data.append([a,"airport"])
    for file in glob.glob(filepath03):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(2)
        train_data.append([a,"basketball_diamond"])
    for file in glob.glob(filepath04):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(3)
        train_data.append([a,"basketball_court"])
    for file in glob.glob(filepath05):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(4)
        train_data.append([a,"beach"])
    for file in glob.glob(filepath06):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(5)
        train_data.append([a,"bridge"])
    for file in glob.glob(filepath07):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(6)
        train_data.append([a,"chaparral"])
    for file in glob.glob(filepath08):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(7)
        train_data.append([a,"church"])
    for file in glob.glob(filepath09):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(8)
        train_data.append([a,"circular_farmland"])
    for file in glob.glob(filepath010):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(9)
        train_data.append([a,"commercial_area"])
    for file in glob.glob(filepath011):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(10)
        train_data.append([a,"dense_residential"])
    for file in glob.glob(filepath012):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(11)
        train_data.append([a,"desert"])
    for file in glob.glob(filepath013):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(12)
        train_data.append([a,"golf_course"])
    for file in glob.glob(filepath014):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(13)
        train_data.append([a,"ground_track_field"])
    for file in glob.glob(filepath015):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(14)
        train_data.append([a,"harbor"])
    for file in glob.glob(filepath016):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(15)
        train_data.append([a,"industrial_area"])
    for file in glob.glob(filepath017):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(16)
        train_data.append([a,"intersection"])
    for file in glob.glob(filepath018):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(17)
        train_data.append([a,"forest"])
    for file in glob.glob(filepath019):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(18)
        train_data.append([a,"freeway"])
    for file in glob.glob(filepath020):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(19)
        train_data.append([a,"island"])
    for file in glob.glob(filepath021):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(20)
        train_data.append([a,"lake"])
    for file in glob.glob(filepath022):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(21)
        train_data.append([a,"meadow"])
    for file in glob.glob(filepath023):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(22)
        train_data.append([a,"medium_residential"])
    for file in glob.glob(filepath024):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(23)
        train_data.append([a,"mobile_home_park"])
    for file in glob.glob(filepath025):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(24)
        train_data.append([a,"mountain"])
    for file in glob.glob(filepath026):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(25)
        train_data.append([a,"overpass"])
    for file in glob.glob(filepath027):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(26)
        train_data.append([a,"parking_lot"])
    for file in glob.glob(filepath028):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(27)
        train_data.append([a,"railway"])
    for file in glob.glob(filepath029):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(28)
        train_data.append([a,"rectangular_farmland"])
    for file in glob.glob(filepath030):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(29)
        train_data.append([a,"round_about"])
    for file in glob.glob(filepath031):
        a=cv2.imread(file)
        # a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        a=cv2.resize(a,(64,64))
        x_train.append(a)
        y_train.append(30)
        train_data.append([a,"runway"])
    
    return np.asarray(x_train),np.asarray(y_train)


def identity_block(X,f,filters,stage,block):
    conv_name_base='res'+str(stage)+block+'_branch'
    bn_name_base='bn'+str(stage)+block+'_branch'
    F1,F2,F3=filters
    X_shortcut=X

    X=Conv2D(filters=F1,kernel_size=(1,1),strides=(1,1),padding='valid',name=
    conv_name_base+'2a',kernel_initializer=glorot_uniform(seed=0))(X)
    X=BatchNormalization(axis=3,name=bn_name_base+'2a')(X)
    X=Activation('relu')(X)
    X=Conv2D(filters=F2,kernel_size=(f,f),strides=(1,1),padding='same',name=
    conv_name_base+'2b',kernel_initializer=glorot_uniform(seed=0))(X)
    X=BatchNormalization(axis=3,name=bn_name_base+'2b')(X)
    X=Activation('relu')(X)

    X=Conv2D(filters=F3,kernel_size=(1,1),padding='valid',name=
    conv_name_base+'2c',kernel_initializer=glorot_uniform(seed=0))(X)
    X=Add()([X,X_shortcut])
    X=Activation('relu')(X)
    return X


def convolutional_block(X,f,filters,stage,block,s=2):
    conv_name_base='res'+str(stage)+block+'_branch'
    bn_name_base='bn'+str(stage)+block+'_branch'

    F1,F2,F3=filters
    X_shortcut=X

    X=Conv2D(filters=F1,kernel_size=(1,1),strides=(s,s),padding='valid',name=
    conv_name_base+'2a',kernel_initializer=glorot_uniform(seed=0))(X)
    X=BatchNormalization(axis=3,name=bn_name_base+'2a')(X)
    X=Activation('relu')(X)


    X=Conv2D(filters=F2,kernel_size=(f,f),strides=(1,1),padding='same',name=
    conv_name_base+'2b',kernel_initializer=glorot_uniform(seed=0))(X)
    X=BatchNormalization(axis=3,name=bn_name_base+'2b')(X)
    X=Activation('relu')(X)


    X=Conv2D(filters=F3,kernel_size=(1,1),strides=(1,1),padding='valid',name=
    conv_name_base+'2c',kernel_initializer=glorot_uniform(seed=0))(X)
    X=BatchNormalization(axis=3,name=bn_name_base+'2c')(X)
    X=Activation('relu')(X)


    X_shortcut=Conv2D(filters=F3,kernel_size=(1,1),strides=(s,s),padding='valid',name=
    conv_name_base+'1',kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut=BatchNormalization(axis=3,name=bn_name_base+'1')(X_shortcut)

    X=Add()([X,X_shortcut])
    X=Activation('relu')(X)
    return X
    


def ResNet50(input_shape=(64,64,3),classes=31):
    X_input=Input(input_shape)

    X=ZeroPadding2D((3,3))(X_input)

    X=Conv2D(64,(7,7),strides=(2,2),name='conv1',kernel_initializer=glorot_uniform(seed=0))(X)
    X=BatchNormalization(axis=3,name='bn_conv1')(X)
    X=Activation('relu')(X)
    X=MaxPooling2D((3,3),strides=(2,2))(X)


    X=convolutional_block(X,f=3,filters=[64,64,256],stage=2,block='a',s=1)
    X=identity_block(X,3,[64,64,256],stage=2,block='b')
    X=identity_block(X,3,[64,64,256],stage=2,block='c')


    X=convolutional_block(X,f=3,filters=[128,128,512],stage=3,block='a',s=2)
    X=identity_block(X,3,[128,128,512],stage=3,block='b')
    X=identity_block(X,3,[128,128,512],stage=3,block='c')
    X=identity_block(X,3,[128,128,512],stage=3,block='d')


    X=convolutional_block(X,f=3,filters=[256,256,1024],stage=4,block='a',s=2)
    X=identity_block(X,3,[256,256,1024],stage=4,block='b')
    X=identity_block(X,3,[256,256,1024],stage=4,block='c')
    X=identity_block(X,3,[256,256,1024],stage=4,block='d')
    X=identity_block(X,3,[256,256,1024],stage=4,block='e')
    X=identity_block(X,3,[256,256,1024],stage=4,block='f')


    X=convolutional_block(X,f=3,filters=[512,512,2048],stage=5,block='a',s=1)
    X=identity_block(X,3,[512,512,2048],stage=5,block='b')
    X=identity_block(X,3,[512,512,2048],stage=5,block='c')

    X=AveragePooling2D(pool_size=(2,2),padding='same')(X)
    X=Flatten()(X)
    X=Dense(classes,activation='softmax',name='fc'+str(classes),kernel_initializer=glorot_uniform(seed=0))(X)

    model=Model(inputs=X_input,outputs=X,name='ResNet50')
    return model





def main():
    x_train,y_train=read_train_file()
    x_test,y_test=read_test_file()
    # print(type(y_train))
    print(x_test.shape,y_test.shape)
    print(x_train.shape)


    x_train=x_train.astype('float32')
    x_test=x_test.astype('float32')

    x_train=x_train/255
    x_test=x_test/255

    y_train=to_categorical(y_train,31)
    y_test=to_categorical(y_test,31)


    model=ResNet50(input_shape=(64,64,3),classes=31)
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit(x_train,y_train,epochs=2,batch_size=32)
    preds=model.evaluate(x_test,y_test)
    print("Loss = " + str(preds[0]))
    print("Test Accuracy = "+ str(preds[1]))

    model.summary()


    
        
        




if __name__ == "__main__":
    main()
