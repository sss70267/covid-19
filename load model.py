from  tensorflow.keras.models import  load_model
from imutils import  paths
import  numpy as np
import  cv2,os

imagepath=list(paths.list_images('./demoset'))
data=[]
filenames=[]
print(imagepath)
for imagepath1 in imagepath:
    print(imagepath1)
    try:
        image = cv2.imread(imagepath1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        filenames0 = imagepath1.split(os.path.sep)[-1]
        filenames.append(filenames0)
        print('data代表資料')
        data.append(image)
    except:
        print('檔案讀取異常')

print('圖檔大小差異很大，進行標準化處理')
data = np.array(data) / 255.0
model=load_model('covid.h5')
predidxs=model.predict(data,batch_size=9)
print(predidxs)
predidxs=np.argmax(predidxs,axis=1)

index1=0
for i in predidxs:
    if i==0:
        print(filenames[index1],end=' ')
        print('covid')
    else:
        print(filenames[index1],end=' ')
        print('normal')
    index1+=1