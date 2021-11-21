################################################################################
#                                                                              #
#  File name : main.py                                                         #
#  Version   : Python 3.8.3rc1 64bit (Tenserflow: 2.3.0) (Keras: 2.4.3)        #
#  Author    : Neel Zadafiya                                                   #
#  StudentId : 1115533                                                         #
#  Purpose   : Emotion classification using EEG dataset                        #
#  Note      : Please use appropriate version of libs to avoid runtime error   #
#                                                                              #
################################################################################

#Import Libraries
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, KFold
from keras.utils import to_categorical
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, LSTM
from keras import Sequential
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler
from keras.layers.embeddings import Embedding
from sklearn.tree import DecisionTreeClassifier
from elm import ELMClassifier
from matplotlib import pyplot as plt

#Define lists to store data

column_delta    = list()    #Lists to store column numbers of different channels
column_theta    = list()
column_alpha    = list()
column_beta     = list()
column_gamma    = list()

#Lists to contain data abour channels
train_data_channel = list()
test_data_channel = list()

#Lists to store accuracies
knn_acc = list()
svm_acc = list()
ann_acc = list()
elm_acc = list()
all_acc = list()

#Generate column numbers for each channel
for i in range(62):
    column_delta.append(i * 5 + 0)
    column_theta.append(i * 5 + 1)
    column_alpha.append(i * 5 + 2)
    column_beta.append(i * 5 + 3)
    column_gamma.append(i * 5 + 4)

#Define arrays to load data
train_data = 0
train_label = 0
test_data = 0
test_label = 0


#================================ Preprocess ==================================#


#Load train data
with open('dataset/train', 'rb') as fo:         #Open file
    d = pickle.load(fo, encoding='bytes')       #Load pickle data
    train_data = d['data']                      #Load train data
    train_label = d['label']                    #Load train label

#Load test data
with open('dataset/test', 'rb') as fo:          #Open file
    d = pickle.load(fo, encoding='bytes')       #Load pickle data
    test_data = d['data']                       #Load test data
    test_label = d['label']                     #Load test label

#Merge train and test data in order to scale them
data = np.concatenate([train_data,test_data])

#Initialize scaler and transform data
scaler = MinMaxScaler()
scaler.fit(data)
data = scaler.transform(data)

#Split data into training and testing categories
train_data = data[:84420]
test_data = data[84420:]

#Extract data for channel delta
train_data_delta = train_data[:,column_delta]
train_data_channel.append(train_data_delta)
test_data_delta = test_data[:,column_delta]
test_data_channel.append(test_data_delta)

#Extract data for channel theta
train_data_theta = train_data[:,column_theta]
train_data_channel.append(train_data_theta)
test_data_theta = test_data[:,column_theta]
test_data_channel.append(test_data_theta)

#Extract data for channel alpha
train_data_alpha = train_data[:,column_alpha]
train_data_channel.append(train_data_alpha)
test_data_alpha = test_data[:,column_alpha]
test_data_channel.append(test_data_alpha)

#Extract data for channel beta
train_data_beta = train_data[:,column_beta]
train_data_channel.append(train_data_beta)
test_data_beta = test_data[:,column_beta]
test_data_channel.append(test_data_beta)

#Extract data for channel gamma
train_data_gamma = train_data[:,column_gamma]
train_data_channel.append(train_data_gamma)
test_data_gamma = test_data[:,column_gamma]
test_data_channel.append(test_data_gamma)


#=================================== KNN ======================================#


#Run models for individual channels
for i in range(5):

    knn = KNeighborsClassifier(n_neighbors= 5)              #Initialize model
    knn.fit(train_data_channel[i], train_label)             #Fit model
    pred_label = knn.predict(test_data_channel[i])          #Predict data
    knn_acc.append(accuracy_score(pred_label, test_label))  #Calculate accuracy
    print('KNN','Channel',i,'Accuracy :',knn_acc[i])        #Print results

#Run model for combined channels 
knn = KNeighborsClassifier(n_neighbors= 5)                  #Initialize model
knn.fit(train_data, train_label)                            #Fit model
pred_label = knn.predict(test_data)                         #Predict data
all_acc.append(accuracy_score(pred_label, test_label))      #Calculate accuracy
print('KNN','Channel','All','Accuracy :',all_acc[-1])       #Print results


#=================================== SVM ======================================#


#Run models for individual channels
for i in range(5):

    svm = SVC()                                             #Initialize model
    svm.fit(train_data_channel[i], train_label)             #Fit model
    pred_label = svm.predict(test_data_channel[i])          #Predict data
    svm_acc.append(accuracy_score(pred_label, test_label))  #Calculate accuracy
    print('SVM','Channel',i,'Accuracy :',svm_acc[i])        #Print results    

#Run model for combined channels     
svm = SVC()                                                 #Initialize model
svm.fit(train_data, train_label)                            #Fit model
pred_label = svm.predict(test_data)                         #Predict data
all_acc.append(accuracy_score(pred_label, test_label))      #Calculate accuracy
print('SVM','Channel','All','Accuracy :',all_acc[-1])       #Print results


#=================================== ANN ======================================#


#Lists for label translation
new_train_label = list()
new_test_label = list()

#Translate label from decimal to unary
for i in train_label:
    if i == 0:
        new_train_label.append([1,0,0])
    if i == 1:
        new_train_label.append([0,1,0])
    if i == 2:
        new_train_label.append([0,0,1])

#Translate label from decimal to unary        
for i in test_label:
    if i == 0:
        new_test_label.append([1,0,0])
    if i == 1:
        new_test_label.append([0,1,0])
    if i == 2:
        new_test_label.append([0,0,1])

#Convert list into numpy array        
new_train_label = np.array(new_train_label)
new_test_label = np.array(new_test_label)

#Run models for individual channels
for i in range(5):

    model = Sequential()                                        #Initialize model
    model.add(Dense(128, input_dim = 62, activation = 'relu'))  #Add dense layers
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(3, activation = 'softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])      #Compile model
    model.fit(train_data_channel[i], new_train_label, epochs = 10, batch_size = 128)            #Fit model
    _,accuracy = model.evaluate(test_data_channel[i], new_test_label)                           #Calculate accuracy
    ann_acc.append(accuracy)
    print('ANN','Channel',i,'Accuracy :',ann_acc[i])                                            #Print Results    

#Run model for combined channels    
model = Sequential()                                            #Initialize model
model.add(Dense(128, input_dim = 310, activation = 'relu'))     #Add dense layers
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation = 'softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])          #Compile model
model.fit(train_data, new_train_label, epochs = 10, batch_size = 128)                           #Fit model
_,accuracy = model.evaluate(test_data, new_test_label)                                          #Calculate accuracy
all_acc.append(accuracy)
print('ANN','Channel','All','Accuracy :',all_acc[-1])                                           #Print Results   


#=================================== ELM ======================================#


#Run models for individual channels
for i in range(5):

    elm = ELMClassifier(alpha=0.0001)                       #Initialize model
    elm.fit(train_data_channel[i],train_label)              #Fit model
    pred_label = elm.predict(test_data_channel[i])          #Predict data
    elm_acc.append(accuracy_score(pred_label, test_label))  #Calculate accuracy
    print('ELM','Channel',i,'Accuracy :',elm_acc[i])        #Print results

#Run model for combined channels    
elm = ELMClassifier(alpha=0.0001)                           #Initialize model
elm.fit(train_data,train_label)                             #Fit model
pred_label = elm.predict(test_data)                         #Predict data
all_acc.append(accuracy_score(pred_label, test_label))      #Calculate accuracy
print('ELM','Channel','All','Accuracy :',all_acc[-1])       #Print results


#=============================== Random Forest ================================#


#Run model for combined channels
clf = RandomForestClassifier(n_estimators = 200, max_depth = None, random_state = 100)  #Initialize model
clf.fit(train_data, train_label)                                                        #Fit model
pred_label = clf.predict(test_data)                                                     #Predict data        
all_acc.append(accuracy_score(pred_label, test_label))                                  #Calculate accuracy
print('RF','Channel','All','Accuracy :',all_acc[-1])                                    #Print results


#================================ Print & Plot ================================#

print('Accuracy from [KNN,SVM,ANN,ELM,RF] =',all_acc)  #Print accuracy from all classifiers for combined channels

#Plot list

a = [0,1,2,3,4]

#Plot graphs for individual channel
for i in range(5):

    a[i] = plt.subplot(2,3,i+1) 
    a[i].bar(['KNN\n'+str(round(knn_acc[i]*100,2)),
            'SVM\n'+str(round(svm_acc[i]*100,2)),
            'ANN\n'+str(round(ann_acc[i]*100,2)),
            'ELM\n'+str(round(elm_acc[i]*100,2))],
            [knn_acc[i]*100,svm_acc[i]*100,ann_acc[i]*100,elm_acc[i]*100],width = 0.6)
    a[i].set_ylabel('Accuracy %')
    
    plt.ylim(0,100)
    
a[0].set_title('Plot 1: Channel Delta')     #Set titles for all graphs
a[1].set_title('Plot 2: Channel Theta')
a[2].set_title('Plot 3: Channel Alpha')
a[3].set_title('Plot 4: Channel Beta')
a[4].set_title('Plot 5: Channel Gamma')

#Plot graph for combined channels
a = plt.subplot(2,3,6)
a.bar(['KNN\n'+str(round(all_acc[0]*100,2)),
        'SVM\n'+str(round(all_acc[1]*100,2)),
        'ANN\n'+str(round(all_acc[2]*100,2)),
        'ELM\n'+str(round(all_acc[3]*100,2)),
        'RF\n'+str(round(all_acc[4]*100,2))],
[100 * i for i in all_acc],width = 0.6)
a.set_ylabel('Accuracy %')

a.set_title('Plot 6: All Channels')         #Set title for last graph
plt.ylim(0,100)

plt.show()                                  #Show graph    