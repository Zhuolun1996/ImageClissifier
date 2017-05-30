from keras.models import load_model
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils


model1 = load_model('imageModelVGG64.h5')
#model2 = load_model('imageModelVGG64.h5')

train_X=np.load('dataSet64.npy')
train_y=np.load('label.npy')
encoder = LabelEncoder()
y_train = encoder.fit_transform(train_y)

temp=[]
for i in range(1,train_y.shape[0]):
    if train_y[i]!=train_y[i-1]:
        temp.append(i)
print(temp)
temp1=model1.predict(train_X)
#temp2=model2.predict(train_X)
#result = np_utils.probas_to_classes(result)
result=[]
#for i in range(len(temp1)):
#    for j in range(len(temp1[i])):
#        temp1[i][j]=max(temp1[i][j],temp2[i][j])
for item in temp1:
    result.append(np.argmax(item))

for item in temp:
    print(train_y[item-1],'->',y_train[item-1])
print(train_y[-1],'->',y_train[-1])

_sum={}
for item in result[0:temp[0]]:
    if item in _sum.keys():
        _sum[item]+=1
    else:
        _sum[item]=1
print(max(_sum.values())/temp[0])
print(_sum,temp[0])

_sum={}
for item in result[temp[0]:temp[1]]:
    if item in _sum.keys():
        _sum[item]+=1
    else:
        _sum[item]=1
print(max(_sum.values())/(temp[1]-temp[0]))
print(_sum,temp[1]-temp[0])

_sum={}
for item in result[temp[1]:temp[2]]:
    if item in _sum.keys():
        _sum[item]+=1
    else:
        _sum[item]=1
print(max(_sum.values())/(temp[2]-temp[1]))
print(_sum,temp[2]-temp[1])

_sum={}
for item in result[temp[2]:temp[3]]:
    if item in _sum.keys():
        _sum[item]+=1
    else:
        _sum[item]=1
print(max(_sum.values())/(temp[3]-temp[2]))
print(_sum,temp[3]-temp[2])

_sum={}
for item in result[temp[3]:temp[4]]:
    if item in _sum.keys():
        _sum[item]+=1
    else:
        _sum[item]=1
print(max(_sum.values())/(temp[4]-temp[3]))
print(_sum,temp[4]-temp[3])

_sum={}
for item in result[temp[4]:temp[5]]:
    if item in _sum.keys():
        _sum[item]+=1
    else:
        _sum[item]=1
print(max(_sum.values())/(temp[5]-temp[4]))
print(_sum,temp[5]-temp[4])

_sum={}
for item in result[temp[5]:temp[6]]:
    if item in _sum.keys():
        _sum[item]+=1
    else:
        _sum[item]=1
print(max(_sum.values())/(temp[6]-temp[5]))
print(_sum,temp[6]-temp[5])

_sum={}
for item in result[temp[6]:temp[7]]:
    if item in _sum.keys():
        _sum[item]+=1
    else:
        _sum[item]=1
print(max(_sum.values())/(temp[7]-temp[6]))
print(_sum,temp[7]-temp[6])

_sum={}
for item in result[temp[7]:temp[8]]:
    if item in _sum.keys():
        _sum[item]+=1
    else:
        _sum[item]=1
print(max(_sum.values())/(temp[8]-temp[7]))
print(_sum,temp[8]-temp[7])

_sum={}
for item in result[temp[8]:]:
    if item in _sum.keys():
        _sum[item]+=1
    else:
        _sum[item]=1
print(max(_sum.values())/(train_X.shape[0]-temp[8]))
print(_sum,(train_X.shape[0]-temp[8]))
