import  os,sys
stderr=sys.stderr
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import  tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.get_logger().setLevel('ERROR')
print('進行tensorflow 2.x  Windows環境的調整')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
print('進行tensorflow 2.x  Mac環境的調整')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
print('完成tensorflow 2.x  設定調整')
print('tensorflow版本:',tf.__version__)
print('python版本:',sys.version)

from imutils import paths
from sklearn.model_selection import train_test_split
from  tensorflow.keras.applications import  vgg16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import  to_categorical
from  sklearn.preprocessing import  LabelEncoder
from tensorflow.keras.layers import  Input,Dropout
from tensorflow.keras.layers import  Flatten,Dense,AveragePooling2D
from tensorflow.keras.models import  Model
import numpy as np
import cv2
from tensorflow.keras.optimizers import  Adam
import matplotlib.pyplot as plt
from sklearn.metrics import  classification_report
from  sklearn.metrics import  confusion_matrix

imagepath = list(paths.list_images('./dataset'))
#print(imagepath)
data = []
labels = []
for imagepath1 in imagepath:
    #print(imagepath1)
    try:
        image = cv2.imread(imagepath1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        temp1 = imagepath1.split(os.path.sep) #分為3個部分
        label = temp1[1]

        labels.append(label)
        data.append(image)
    except:
        print('讀取異常')
#print('-------------------data----------------------------')
data = np.array(data)/255
#print(data)
#print('-------------------labels----------------------------')
labels = np.array(labels)
#print(labels#)#

lb=LabelEncoder()
labels1=lb.fit_transform(labels)

labels2=to_categorical(labels1)
(trainX,testX,trainY,testY)=train_test_split(data,labels2,test_size=0.2,random_state=42)
traingAug=ImageDataGenerator(rotation_range=15,fill_mode='nearest')

baseModel=vgg16.VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)
headModel=baseModel.output
headModel=AveragePooling2D(pool_size=(4,4))(headModel)
headModel=Flatten(name='flatten')(headModel)  #平坦層，將資料轉換為一維架構
headModel=Dense(64,activation='relu')(headModel) #完全連接層1
headModel=Dropout(0.5)(headModel)
headModel=Dense(2,activation='softmax')(headModel)
model=Model(inputs=baseModel.input,outputs=headModel)
for layer in baseModel.layers:
    layer.trainable=False
print('顯示模型結構')
model.summary()
tf.keras.utils.plot_model(model,to_file='covid19.png')

print('設定參數，後續可以調整')
INIT_LR=1e-3
EPOCHS=25
BS=6

print('最佳化的規劃')
opt=Adam(learning_rate = INIT_LR,decay = INIT_LR/EPOCHS)

print('編譯')
model.compile(
    loss = 'binary_crossentropy',optimizer = opt,metrics=['accuracy']
)

print('訓練')
print('steps_per_epoch:',len(trainX)//BS)
print('validation_steps:',len(testX)//BS)

history = model.fit(
    traingAug.flow(trainX,trainY,batch_size=BS),
    steps_per_epoch=len(trainX)//BS,
    validation_data=(testX,testY),
    validation_steps=len(testX)//BS,
    epochs=EPOCHS
)

x1=history.history['accuracy']
x2=history.history['val_accuracy']
x3=history.history['loss']
x4=history.history['val_loss']
plt.plot(x1,c='red')
plt.plot(x2,c='green')
plt.plot(x3,c='blue')
plt.plot(x4,c='yellow')
plt.grid()
plt.show()


print('以測試資料進行預估，藉此評估模型效果')
predidxs = model.predict(testX,batch_size=8)
predidxs=np.argmax(predidxs,axis=1)
print(predidxs)
testY=np.argmax(testY,axis=1)
print(testY)

print(classification_report(testY,predidxs))
print(confusion_matrix(testY,predidxs))
print('模型儲存')
model.save('covid.h5')