#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np


# ### load data

# In[7]:


dataset = 'F-dataset' # B-dataset C-dataset F-dataset
drdi = pd.read_csv('./data/'+dataset+'/DrDiNum.csv', header = None)#, sep='\t', header = None)
drpr = pd.read_csv('./data/'+dataset+'/DrPrNum.csv', header = None)#, sep='\t', header = None)
dipr = pd.read_csv('./data/'+dataset+'/DiPrNum.csv', header = None)#, sep='\t', header = None)

# drdr = pd.read_csv('./data/'+dataset+'/DrDrNum.csv', header = None)
# didi = pd.read_csv('./data/'+dataset+'/DiDiNum.csv', header = None)

drug = drdi# pd.concat([drdi,drpr]) # pd.concat([drdi,drpr,drdr])# pd.concat([drdi,drpr,drdr])# pd.concat([drdi,drdr])# 
disease = drdi.rename(columns={0: 1,1:0})#pd.concat([drdi.rename(columns={0: 1,1:0}),dipr]) # pd.concat([drdi.rename(columns={0: 1,1:0}),dipr,didi])  # pd.concat([drdi.rename(columns={0: 1,1:0}),didi]) #  # 
protein = pd.concat([drpr[[1,0]],dipr[[1,0]]])


# In[8]:


allnode= drdi#pd.concat([drdi,drpr,dipr])
max_node = max([max(allnode[0]),max(allnode[1])])
max_node


# ### search matepath

# In[9]:


def positive_sampler(path):

    pos_u,pos_v=[],[]
    for i in range(len(path)):
        if len(path)==1:
            continue
        u=path[i]
        v=np.concatenate([path[max(i-window,0):i],path[i+1:i+window+1]],axis=0)
        pos_u.extend([u]*len(v))
        pos_v.extend(v)
    return pos_u,pos_v    
def get_negative_ratio(metapath):

    node_frequency=dict()
    sentence_count,node_count=0,0
    for path in metapath:
        for node in path:
            node_frequency[node]=node_frequency.get(node,0)+1
            node_count+=1
    pow_frequency=np.array(list(map(lambda x:x[-1],sorted(node_frequency.items(),key=lambda asd:asd[0]))))**0.75
    node_pow=np.sum(pow_frequency)
    ratio=pow_frequency/node_pow
    return ratio
def negative_sampler(path,ratio,nodes):

    negtives_size=5
    negatives=[]
    while len(negatives)<5:
        temp=np.random.choice(nodes, size=negtives_size-len(negatives), replace=False, p=ratio)
        negatives.extend([node for node in temp if node not in path])
    return negatives


# In[10]:


from tqdm import tqdm
import numpy as np
import torch
def create_node2node_dict(graph):
    src_dst={}
    for src,dst in zip(graph[:,0],graph[:,1]):
        src,dst=src.item(),dst.item()
        if src not in src_dst.keys():
            src_dst[src]=[]
        src_dst[src].append(dst)
    return src_dst
window=10
metapaths=[]
num_walks=10
walk_len= 20 

# 1; 'drug','disease','drug'
# 2: 'drug','protein','disease'   
# 3: 'drug','disease','drug','protein','drug'

# 4 'drug','drug','disease','disease','drug'

metapath_type=['drug','disease','drug']

drug_graph = drug.values
disease_graph = disease.values
protein_graph = protein.values

weights_coauthor = 1
weights_cotitle = 1
weights_covenue = 1

edge_per_graph={}
edge_per_graph['drug']=create_node2node_dict(drug_graph)
edge_per_graph['disease']=create_node2node_dict(disease_graph)
edge_per_graph['protein']=create_node2node_dict(protein_graph)
weights_all_graph={'drug':weights_coauthor,'disease':weights_cotitle,'protein':weights_covenue}
 
def Is_isolate(node):
    for rel in metapath_type:
        if node in edge_per_graph[rel].keys():
            return 0
    return 1
for walk in tqdm(range(num_walks)):
    for cur_node in list(range(max_node+1)):
        stop=0
        path=[]
        path.append(cur_node)
        while len(path)<walk_len and stop==0:
            for rel in metapath_type:
                if len(path)==walk_len or Is_isolate(cur_node):
                    stop=1
                    break
                if edge_per_graph[rel].get(cur_node,-1)==-1:
                    continue
                    
                cand_nodes=edge_per_graph[rel][cur_node]
                cur_node=np.random.choice(cand_nodes,size=1)[0] #,p=weighted_ratio
                path.append(cur_node)
        metapaths.append(path)


# In[11]:


pd.DataFrame(metapaths).to_csv('./data/'+dataset+'/metapaths3.txt',sep='\t',header=0,index=0,mode='a')


# In[ ]:


# pd.DataFrame(metapaths).to_csv('./data/'+dataset+'/metapaths1.txt',sep='\t',header=0,index=0)


# ### check generated metapath2vec.txt

# In[12]:


dataset = 'B-dataset' # C-dataset F-dataset
feature = pd.read_csv('./data/'+dataset+'/metapath2vec1.txt',sep=' ', header=None, skiprows=2) 
feature


# In[13]:


feature.dropna()


# In[14]:


feature = feature.sort_values(0,ascending=True).dropna(axis=1)
feature


# In[15]:


feature.drop_duplicates(0,'first',inplace=True)
feature.set_index([0],inplace=True)
feature


# ### generated drug similarity  disease similarity

# In[7]:



import csv
import math
import random
import xlrd
from numpy import *


def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:  
        SaveList.append(row)
    return

def storFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return


OriginalData = []
ReadMyCsv(OriginalData, "./data/"+dataset+"/DrDiNum.csv")#drug-disease-whole.csv   DrDiNum.csv
print(len(OriginalData))

LncDisease = []
counter = 0
while counter < len(OriginalData):
    Pair = []
    Pair.append(OriginalData[counter][0])
    Pair.append(OriginalData[counter][1])
    LncDisease.append(Pair)
    counter = counter + 1
print('LncDisease的长度', len(LncDisease))
print('OriginalData的长度', len(OriginalData))

AllDisease = []
counter1 = 0
while counter1 < len(OriginalData): 
    counter2 = 0
    flag = 0
    while counter2 < len(AllDisease):  
        if OriginalData[counter1][1] != AllDisease[counter2]:
            counter2 = counter2 + 1
        elif OriginalData[counter1][1] == AllDisease[counter2]:
            flag = 1
            counter2 = counter2 + 1
    if flag == 0:
        AllDisease.append(OriginalData[counter1][1])
    counter1 = counter1 + 1
print('len(AllDisease)', len(AllDisease))

AllDRUG = []
counter1 = 0
while counter1 < len(OriginalData): 
    counter2 = 0
    flag = 0
    while counter2 < len(AllDRUG): 
        if OriginalData[counter1][0] != AllDRUG[counter2]:
            counter2 = counter2 + 1
        elif OriginalData[counter1][0] == AllDRUG[counter2]:
            flag = 1
            break
    if flag == 0:
        AllDRUG.append(OriginalData[counter1][0])
    counter1 = counter1 + 1
# print('AllRNA', AllRNA)
print('len(AllDRUG)', len(AllDRUG))

DiseaseAndDrugBinary = []
counter = 0
while counter < len(AllDisease):
    row = []
    counter1 = 0
    while counter1 < len(AllDRUG):
        row.append(0)
        counter1 = counter1 + 1
    DiseaseAndDrugBinary.append(row)
    counter = counter + 1

print('len(LncDisease)', len(LncDisease))
counter = 0
while counter < len(LncDisease):
    DN = LncDisease[counter][1]
    RN = LncDisease[counter][0]
    counter1 = 0
    while counter1 < len(AllDisease):
        if AllDisease[counter1] == DN:
            counter2 = 0
            while counter2 < len(AllDRUG):
                if AllDRUG[counter2] == RN:
                    DiseaseAndDrugBinary[counter1][counter2] = 1
                    break
                counter2 = counter2 + 1
            break
        counter1 = counter1 + 1
    counter = counter + 1
print('len(DiseaseAndDrugBinary)', len(DiseaseAndDrugBinary))

# disease的文本挖掘相似矩阵
if dataset =='B-dataset':
    dis_sim = pd.read_csv(open('./data/'+dataset+'/disease相似性矩阵.txt'),sep='\t',index_col=0)
    txtSimilarity = dis_sim.values.tolist()
    print('len(txtSimilarity)',len(txtSimilarity))
    print('len(txtSimilarity[1])',len(txtSimilarity[1]))
else: 
    lines = [line.strip().split() for line in open("./data/"+dataset+"/disease相似性矩阵.txt")]
    txtSimilarity = []
    i = 0
    for dis in lines:
        i = i + 1
        if i == 1:
            continue
        txtSimilarity.append(dis[1:])
    print('len(txtSimilarity)',len(txtSimilarity))
    print('len(txtSimilarity[1])',len(txtSimilarity[1]))

# drug的文本挖掘相似矩阵
if dataset =='B-dataset':
    drug_sim = pd.read_csv(open('./data/'+dataset+'/drug相似性矩阵.txt'),sep='\t',index_col=0)
    drugtxtSimilarity = drug_sim.values.tolist()
    print('len(drugtxtSimilarity)',len(drugtxtSimilarity))
    print('len(drugtxtSimilarity[1]',len(drugtxtSimilarity[1]))
else:  
    lines = [line.strip().split() for line in open("./data/"+dataset+"/drug相似性矩阵.txt")]
    drugtxtSimilarity = []
    i = 0
    for dis in lines:
        i = i + 1
        if i == 1:
            continue
        drugtxtSimilarity.append(dis[1:])
    print('len(drugtxtSimilarity)',len(drugtxtSimilarity))
    print('len(drugtxtSimilarity[1]',len(drugtxtSimilarity[1]))


counter1 = 0
sum1 = 0
while counter1 < (len(AllDisease)):
    counter2 = 0
    while counter2 < (len(AllDRUG)):
        sum1 = sum1 + pow((DiseaseAndDrugBinary[counter1][counter2]), 2)
        counter2 = counter2 + 1
    counter1 = counter1 + 1
print('sum1=', sum1)
Ak = sum1
Nd = len(AllDisease)
rdpie = 0.5
rd = rdpie * Nd / Ak
print('disease rd', rd)
# 生成DiseaseGaussian
DiseaseGaussian = []
counter1 = 0
while counter1 < len(AllDisease):
    counter2 = 0
    DiseaseGaussianRow = []
    while counter2 < len(AllDisease):
        AiMinusBj = 0
        sum2 = 0
        counter3 = 0
        AsimilarityB = 0
        while counter3 < len(AllDRUG):
            sum2 = pow((DiseaseAndDrugBinary[counter1][counter3] - DiseaseAndDrugBinary[counter2][counter3]), 2)
            AiMinusBj = AiMinusBj + sum2
            counter3 = counter3 + 1
        AsimilarityB = math.exp(- (AiMinusBj/rd))
        DiseaseGaussianRow.append(AsimilarityB)
        counter2 = counter2 + 1
    DiseaseGaussian.append(DiseaseGaussianRow)
    counter1 = counter1 + 1
print('len(DiseaseGaussian)', len(DiseaseGaussian))
print('len(DiseaseGaussian[0])', len(DiseaseGaussian[0]))
# 构建Drugaussian
from numpy import *
MDiseaseAndDrugBinary = np.array(DiseaseAndDrugBinary)    
DRUGAndDiseaseBinary = MDiseaseAndDrugBinary.T    
DRUGGaussian = []
counter1 = 0
sum1 = 0
while counter1 < (len(AllDRUG)):    
    counter2 = 0
    while counter2 < (len(AllDisease)):     
        sum1 = sum1 + pow((DRUGAndDiseaseBinary[counter1][counter2]), 2)
        counter2 = counter2 + 1
    counter1 = counter1 + 1
print('sum1=', sum1)
Ak = sum1
Nm = len(AllDRUG)
rdpie = 0.5
rd = rdpie * Nm / Ak
print('DRUG rd', rd)
# 生成DRUGGaussian
counter1 = 0
while counter1 < len(AllDRUG):   
    counter2 = 0
    DRUGGaussianRow = []
    while counter2 < len(AllDRUG):   
        AiMinusBj = 0
        sum2 = 0
        counter3 = 0
        AsimilarityB = 0
        while counter3 < len(AllDisease):   
            sum2 = pow((DRUGAndDiseaseBinary[counter1][counter3] - DRUGAndDiseaseBinary[counter2][counter3]), 2)
            AiMinusBj = AiMinusBj + sum2
            counter3 = counter3 + 1
        AsimilarityB = math.exp(- (AiMinusBj/rd))
        DRUGGaussianRow.append(AsimilarityB)
        counter2 = counter2 + 1
    DRUGGaussian.append(DRUGGaussianRow)
    counter1 = counter1 + 1
print('type(DRUGGaussian)', type(DRUGGaussian))
print('len(DRUGGaussian)', len(DRUGGaussian))
print('len(DRUGGaussian[0])', len(DRUGGaussian[0]))


import random
counter1 = 0    
counter2 = 0    
counterP = 0    
counterN = 0    
PositiveSample = []     

PositiveSample = LncDisease
print('PositiveSample)', len(PositiveSample))
pd.DataFrame(PositiveSample).to_csv('./data/'+dataset+'/PositiveSample1.csv',header=0,index=0)
# storFile(PositiveSample, 'PositiveSample.csv')



NegativeSample = []
counterN = 0
while counterN < len(PositiveSample):                         # 
    counterD = random.randint(0, len(AllDisease)-1)
    counterR = random.randint(0, len(AllDRUG)-1)     
    DiseaseAndRnaPair = []
    DiseaseAndRnaPair.append(AllDRUG[counterR])
    DiseaseAndRnaPair.append(AllDisease[counterD])
    flag1 = 0
    counter = 0
    while counter < len(LncDisease):
        if DiseaseAndRnaPair == LncDisease[counter]:
            flag1 = 1
            break
        counter = counter + 1
    if flag1 == 1:
        continue
    flag2 = 0
    counter1 = 0
    while counter1 < len(NegativeSample):
        if DiseaseAndRnaPair == NegativeSample[counter1]:
            flag2 = 1
            break
        counter1 = counter1 + 1
    if flag2 == 1:
        continue
    if (flag1 == 0 & flag2 == 0):
        NegativePair = []
        NegativePair.append(AllDRUG[counterR])
        NegativePair.append(AllDisease[counterD])
        NegativeSample.append(NegativePair)
        counterN = counterN + 1
print('len(NegativeSample)', len(NegativeSample))
pd.DataFrame(NegativeSample).to_csv('./data/'+dataset+'/NegativeSample1.csv',header=0,index=0)

DiseaseSimilarity = []
counter = 0
while counter < len(AllDisease):
    counter1 = 0
    Row = []
    while counter1 < len(AllDisease):
        # v = float(txtSimilarity[counter][counter1])
        v = float(DiseaseGaussian[counter][counter1])
        if v > 0:
            Row.append(v)
        if v == 0:
            Row.append(txtSimilarity[counter][counter1])
        counter1 = counter1 + 1
    DiseaseSimilarity.append(Row)
    counter = counter + 1
print('len(DiseaseSimilarity)', len(DiseaseSimilarity))
print('len(DiseaseSimilarity[0)',len(DiseaseSimilarity[0]))
# pd.DataFrame(DiseaseSimilarity).to_csv('./data/'+dataset+'/DiseaseSimilarity.csv',header=0,index=0)


DRUGSimilarity = []
counter = 0
while counter < len(AllDRUG):
    counter1 = 0
    Row = []
    while counter1 < len(AllDRUG):
        # v = float(drugtxtSimilarity[counter][counter1])
        v = float(DRUGGaussian[counter][counter1])
        if v > 0:
            Row.append(v)
        if v == 0:
            Row.append(drugtxtSimilarity[counter][counter1])
        counter1 = counter1 + 1
    DRUGSimilarity.append(Row)
    counter = counter + 1
print('len(DRUGSimilarity)', len(DRUGSimilarity))
print('len(DRUGSimilarity[0)',len(DRUGSimilarity[0]))
# pd.DataFrame(DRUGSimilarity).to_csv('./data/'+dataset+'/DrugSimilarity.csv',header=0,index=0)


AllSample = PositiveSample.copy()
AllSample.extend(NegativeSample)        

# storFile(AllSample, 'AllSample.csv')

# SampleFeature
SampleFeature = []
counter = 0
while counter < len(AllSample):
    counter1 = 0
    while counter1 < len(AllDRUG):
        if AllSample[counter][0] == AllDRUG[counter1]:
            a = []
            counter3 = 0
            # 原本是ALLDrug
            while counter3 < len(AllDRUG):
                v = DRUGSimilarity[counter1][counter3]
                # v = drugtxtSimilarity[counter1][counter3]
                a.append(v)
                counter3 = counter3 + 1
            break
        counter1 = counter1 + 1
    counter2 = 0
    while counter2 < len(AllDisease):
        if AllSample[counter][1] == AllDisease[counter2]:
            b = []
            counter3 = 0
            # 原本是ALLDisease
            while counter3 < len(AllDisease):
                v = DiseaseSimilarity[counter2][counter3]
                # v=txtSimilarity[counter2][counter3]
                b.append(v)
                counter3 = counter3 + 1
            break
        counter2 = counter2 + 1
    a.extend(b)
    SampleFeature.append(a)
    counter = counter + 1
counter1 = 0
storFile(SampleFeature, './data/'+dataset+'/SampleFeature1.csv')
print('SampleFeature',len(SampleFeature))
print('SampleFeature[1]',len(SampleFeature[1]))


# In[8]:


pd.read_csv('./data/'+dataset+'/SampleFeature1.csv',header=None)


# In[7]:


DrugSimilarity = pd.read_csv('./data/'+dataset+'/DrugSimilarity.csv',header=None)
DrugSimilarity


# In[8]:


DrugSimilarity[DrugSimilarity>0] = 1
DrugSimilarity = DrugSimilarity.stack().reset_index().rename(columns={'level_0':'Source','level_1':'Target', 0:'Weight'})
DrugSimilarity = DrugSimilarity[DrugSimilarity['Weight']==1]
DrugSimilarity[['Source','Target']].to_csv('./data/'+dataset+'/DrDrNum.csv',header=0,index=0)
DrugSimilarity


# In[9]:


DiseaseSimilarity = pd.read_csv('./data/'+dataset+'/DiseaseSimilarity.csv',header=None)
DiseaseSimilarity


# In[13]:


DiseaseSimilarity[DiseaseSimilarity>0] = 1
DiseaseSimilarity = DiseaseSimilarity.stack().reset_index().rename(columns={'level_0':'Source','level_1':'Target', 0:'Weight'})
DiseaseSimilarity = DiseaseSimilarity[DiseaseSimilarity['Weight']==1]
DiseaseSimilarity[['Source','Target']].to_csv('./data/'+dataset+'/DiDiNum.csv',header=0,index=0)
DiseaseSimilarity


# ### CNN

# In[9]:


from numpy import *
import keras
from keras.models import Sequential
from keras.layers import BatchNormalization
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import csv
from sklearn.model_selection import train_test_split

def MyCNN(SampleFeature):
    x = np.array(SampleFeature)
    print(x.shape)
    # data = []
    if dataset=='B-dataset':   
        data1 = ones((18416,1), dtype=int)
        data2 = zeros((18416,1))
    elif dataset=='F-dataset':
        data1 = ones((1933,1), dtype=int)
        data2 = zeros((1933,1))
    else:
        data1 = ones((2532,1), dtype=int)
        data2 = zeros((2532,1))    
    y = np.concatenate((data1, data2), axis=0) # data1 
    print(y.shape)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)  

    x_train = x
    y_train = y
    if dataset=='B-dataset':   
        node_sum = 867
    elif dataset=='F-dataset':
        node_sum = 906
    else:
        node_sum = 1072
    x_train = x_train.reshape(-1, 1, node_sum, 1)
    # 970 1136
#     x_test = x_test.reshape(-1, 1, node_sum, 1)
    x = x.reshape(-1, 1, node_sum, 1)
#     print(x_train.shape)
#     print(x_test.shape)
    batch_size = 32  
    epochs = 2
    model = Sequential()  
    return_sequences = True

    model.add(Conv2D(32, (4, 4), strides=(2, 2), activation='relu', padding='same', data_format='channels_last',
                     name='layer1_con1', input_shape=(1, node_sum, 1)))
    # model.add(Conv2D(64, (16, 16), strides=(2, 2), activation='relu', padding='same', data_format='channels_last',
    #                  name='layer1_con2'))
    # model.add(Conv2D(128, (32, 32), strides=(2, 2), activation='relu', padding='same', data_format='channels_last',
    #                  name='layer1_con3'))
    model.add(Dropout(0.5))

    model.add(
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format='channels_last', name='layer1_pool'))


    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(256, activation='relu', name='Dense-2'))  
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid', ))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    # import numpy as np
    model.fit(x_train, y_train, epochs=30, batch_size=10, validation_split=0.1)
    from keras.models import Model
    dense1_layer_model = Model(inputs=model.input, outputs=model.get_layer('Dense-2').output)
    dense1_output = dense1_layer_model.predict(x)
    print(dense1_output.shape)
    print(dense1_output[0])
    return dense1_output


# In[46]:


# dataset='F-dataset'
SampleFeature = pd.read_csv('./data/'+dataset+'/SampleFeature1.csv',header=None)
dense1_output = MyCNN(SampleFeature)
pd.DataFrame(dense1_output).to_csv('./data/'+dataset+'/dense1_output1.csv',header=0,index=0)


# In[2]:


import sys
sys.version


# In[6]:


get_ipython().system('pip install protobuf==3.20.1')


# ### translate data types

# In[24]:


dda = pd.read_csv('./data/'+dataset+'/drug_disease.txt',sep='\t', header=None) 
dda.to_csv('./data/'+dataset+'/DrDiNum.csv',header=0,index=0)
dda


# In[26]:


dda = pd.read_csv('./data/'+dataset+'/protein_disease.txt',sep='\t', header=None) 
dda[[1,0]].to_csv('./data/'+dataset+'/DiPrNum.csv',header=0,index=0)
dda[[1,0]]


# In[27]:


dda = pd.read_csv('./data/'+dataset+'/protein_drug.txt',sep='\t', header=None) 
dda[[1,0]].to_csv('./data/'+dataset+'/DrPrNum.csv',header=0,index=0)
dda[[1,0]]


# In[9]:


from sklearn.metrics import roc_curve,auc
predicted = pd.read_csv('./results/B-dataset/test_auc1.txt',sep=' ',header=None)
fpr, tpr, thresholds = roc_curve(np.array(predicted[0]), predicted[1])
roc_auc = auc(fpr, tpr)
roc_auc


# ### evaluation

# In[26]:


allnode = pd.read_csv('./data/'+dataset+'/AllNode.csv',header=None,names=['index','id'],skiprows=1)
allnode['id'] = allnode['id'].str.lower()
Negative = pd.read_csv('./data/'+dataset+'/PositiveSample.csv', header = None)
Negativenum1 = pd.merge(Negative,allnode,how='left',left_on=0,right_on='id')
Negativenum2 = pd.merge(Negative,allnode,how='left',left_on=1,right_on='id')
Negativenum = pd.concat([Negativenum1[['index']].fillna(0).astype('int64'),Negativenum2[['index']]],axis=1)
Negativenum.to_csv('./data/'+dataset+'/PositiveNum.csv',header=0,index=0)
Negativenum


# In[27]:


allnode = pd.read_csv('./data/'+dataset+'/AllNode.csv',header=None,names=['index','id'],skiprows=1)
allnode['id'] = allnode['id'].str.lower()
Negative = pd.read_csv('./data/'+dataset+'/NegativeSample.csv', header = None)
Negativenum1 = pd.merge(Negative,allnode,how='left',left_on=0,right_on='id')
Negativenum2 = pd.merge(Negative,allnode,how='left',left_on=1,right_on='id')
Negativenum = pd.concat([Negativenum1[['index']].fillna(0).astype('int64'),Negativenum2[['index']]],axis=1)
Negativenum.to_csv('./data/'+dataset+'/NegativeNum.csv',header=0,index=0)
Negativenum


# In[37]:


dataset = 'C-dataset'#'B-dataset' # C-dataset F-dataset
Positive = pd.read_csv('./data/'+dataset+'/PositiveSample1.csv', header = None)# DrDiNum.csv
Negative = pd.read_csv('./data/'+dataset+'/NegativeSample1.csv', header = None)#NegativeNum  AllNegativeSample
#Negative = Negative.sample(n=18416, random_state=18416)
Attribute = pd.read_csv('./data/'+dataset+'/metapath2vec2.txt',sep=' ', header=None, skiprows=2)
Attribute = Attribute.sort_values(0,ascending=True).dropna(axis=1)
Attribute.drop_duplicates(0,'first',inplace=True)
Attribute.set_index([0],inplace=True)
Positive[2] = Positive.apply(lambda x:1 if x[0] < 0 else 1, axis=1)
Negative[2] = Negative.apply(lambda x:0 if x[0] < 0 else 0, axis=1)
result = pd.concat([Positive,Negative]).reset_index(drop=True)
X1 = pd.concat([Attribute.loc[result[0].values.tolist()].reset_index(drop=True),Attribute.loc[result[1].values.tolist()].reset_index(drop=True)],axis=1)
Y = result[2]


# In[38]:


from numpy import *
X2 = pd.read_csv('./data/'+dataset+'/SampleFeature1.csv',header=None)#dense1_output1    SampleFeature1
#NegativeFeature = pd.read_csv('./data/'+dataset+'/Negativefeature.csv',header=None)
#X2 = pd.concat([X2,NegativeFeature]).reset_index(drop=True)
X2.fillna(0,inplace=True)
if dataset=='B-dataset':   
    data1 = ones((1, 18416), dtype=int)
    data2 = zeros((1, 18416))
elif dataset=='F-dataset':
    data1 = ones((1, 1933), dtype=int)
    data2 = zeros((1, 1933))
else:
    data1 = ones((1, 2532), dtype=int)
    data2 = zeros((1, 2532))    
Y = pd.DataFrame(np.concatenate((data1, data2), axis=1).T)

X = pd.concat([X1,X2],axis=1).reset_index(drop=True)


from sklearn.model_selection import  StratifiedKFold,KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from scipy import interp
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score

from src.metrics import *


k_fold = 10
print("%d fold CV"% k_fold)
i=0
skf = StratifiedKFold(n_splits=k_fold,random_state=0, shuffle=True)
for train_index, test_index in skf.split(X,Y):
  
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
    model = RandomForestClassifier(n_estimators=999,n_jobs=-1)#n_estimators=999,n_jobs=-1
    model.fit(np.array(X_train), np.array(Y_train))
    y_score0 = model.predict(np.array(X_test))
    y_score_RandomF = model.predict_proba(np.array(X_test))
    fpr,tpr,thresholds=roc_curve(Y_test,y_score_RandomF[:,1])
    roc_auc=auc(fpr,tpr)
    print("---------------------------------------------")
    print("fold = ", roc_auc)
    print("---------------------------------------------\n")
