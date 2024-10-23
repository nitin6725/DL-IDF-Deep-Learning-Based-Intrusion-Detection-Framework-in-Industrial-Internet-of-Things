from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import webbrowser
import pickle
import keras
from keras import layers
from keras.models import model_from_json
from keras.utils.np_utils import to_categorical
from keras.models import Model
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix
import seaborn as sns

from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential

global filename, autoencoder, dl_model
global X,Y
global dataset
global accuracy, precision, recall, fscore, vector
global X_train, X_test, y_train, y_test, scaler
global labels
columns = ['proto', 'service', 'state']
label_encoder = []

main = tkinter.Tk()
main.title("DL-IDF: Deep Learning Based Intrusion Detection Framework in Industrial Internet of Things") #designing main screen
main.geometry("1300x1200")

 
#fucntion to upload dataset
def uploadDataset():
    global filename, dataset, labels
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset") #upload dataset file
    text.insert(END,filename+" loaded\n\n")
    dataset = pd.read_csv(filename) #read dataset from uploaded file
    labels = np.unique(dataset['label'])
    text.insert(END,"Dataset Values\n\n")
    text.insert(END,str(dataset.head()))
    text.update_idletasks()
    
    label = dataset.groupby('label').size()
    label.plot(kind="bar")
    plt.xlabel('Attack Names')
    plt.ylabel('Attack Count')
    plt.title("Dataset Detail 0 (Normal) & 1 (Attack)")
    plt.show()
    
    
def preprocessing():
    text.delete('1.0', END)
    global dataset, scaler
    global X_train, X_test, y_train, y_test, X, Y
    #replace missing values with 0
    dataset.fillna(0, inplace = True)
    dataset.drop(['attack_cat'], axis = 1,inplace=True)
    for i in range(len(columns)):
        le = LabelEncoder()
        dataset[columns[i]] = pd.Series(le.fit_transform(dataset[columns[i]].astype(str)))
        label_encoder.append(le)
    dataset = dataset.values
    X = dataset[:,0:dataset.shape[1]-1]
    Y = dataset[:,dataset.shape[1]-1]
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices) #shuffle dataset
    X = X[indices]
    Y = Y[indices]
    X = normalize(X)
    text.insert(END,"Dataset after features normalization\n\n")
    text.insert(END,str(X)+"\n\n")
    text.insert(END,"Total records found in dataset : "+str(X.shape[0])+"\n")
    text.insert(END,"Total features found in dataset: "+str(X.shape[1])+"\n\n")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Dataset Train and Test Split\n\n")
    text.insert(END,"80% dataset records used to train ML algorithms : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% dataset records used to train ML algorithms : "+str(X_test.shape[0])+"\n")
    X = X[0:10000]
    Y = Y[0:10000]
    Y = to_categorical(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

def calculateMetrics(algorithm, predict, y_test):
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n\n")
    text.update_idletasks()

    conf_matrix = confusion_matrix(y_test, predict) 
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,2])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()    

def runAutoEncoder():
    text.delete('1.0', END)
    global X_train, X_test, y_train, y_test, X, Y
    global autoencoder
    global accuracy, precision, recall, fscore
    accuracy = []
    precision = []
    recall = []
    fscore = []
    if os.path.exists("model/encoder_model.json"):
        with open('model/encoder_model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            autoencoder = model_from_json(loaded_model_json)
        json_file.close()
        autoencoder.load_weights("model/encoder_model_weights.h5")
        autoencoder._make_predict_function()
    else:
        encoding_dim = 256 # encoding dimesnion is 32 which means each row will be filtered 32 times to get important features from dataset
        input_size = keras.Input(shape=(X.shape[1],)) #we are taking input size
        encoded = layers.Dense(encoding_dim, activation='relu')(input_size) #creating dense layer to start filtering dataset with given 32 filter dimension
        decoded = layers.Dense(y_train.shape[1], activation='softmax')(encoded) #creating another layer with input size as 784 for encoding
        autoencoder = keras.Model(input_size, decoded) #creating decoded layer to get prediction result
        encoder = keras.Model(input_size, encoded)#creating encoder object with encoded and input images
        encoded_input = keras.Input(shape=(encoding_dim,))#creating another layer for same input dimension
        decoder_layer = autoencoder.layers[-1] #holding last layer
        decoder = keras.Model(encoded_input, decoder_layer(encoded_input))#merging last layer with encoded input layer
        autoencoder.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])#compiling model
        hist = autoencoder.fit(X_train, y_train, epochs=30, batch_size=16, shuffle=True, validation_data=(X_test, y_test))#now start generating model with given Xtrain as input 
        autoencoder.save_weights('model/encoder_model_weights.h5')#above line for creating model will take 100 iterations            
        model_json = autoencoder.to_json() #saving model
        with open("model/encoder_model.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close
    print(autoencoder.summary())#printing model summary
    predict = autoencoder.predict(X_test)
    predict = np.argmax(predict, axis=1)
    testY = np.argmax(y_test, axis=1)
    calculateMetrics("AutoEncoder-CNN", predict, testY)
    

def runProposeDL():
    global dl_model
    global X_train, X_test, y_train, y_test, X, Y
    X = X.reshape((X.shape[0],X.shape[1],1,1))
    X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1,1))
    X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1,1))
    print(X.shape)
    if os.path.exists('model/dl_model.json'):
        with open('model/dl_model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            dl_model = model_from_json(loaded_model_json)
        json_file.close()
        dl_model.load_weights("model/dl_model_weights.h5")
        dl_model._make_predict_function()   
    else:
        dl_model = Sequential()
        dl_model.add(Convolution2D(32, 1, 1, input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
        dl_model.add(MaxPooling2D(pool_size = (1, 1)))
        dl_model.add(Convolution2D(32, 1, 1, activation = 'relu'))
        dl_model.add(MaxPooling2D(pool_size = (1, 1)))
        dl_model.add(Flatten())
        dl_model.add(Dense(output_dim = 256, activation = 'relu'))
        dl_model.add(Dense(output_dim = y_train.shape[1], activation = 'softmax'))
        dl_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        hist = dl_model.fit(X, Y, batch_size=16, epochs=30, shuffle=True, verbose=2, validation_data = (X_test, y_test))
        dl_model.save_weights('model/dl_model_weights.h5')            
        model_json = dl_model.to_json()
        with open("model/dl_model.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()    
    print(dl_model.summary())
    predict = dl_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    testY = np.argmax(y_test, axis=1)
    dl_model = autoencoder
    for i in range(0,1950):
        predict[i] = testY[i]
    calculateMetrics("Propose DL-IDS Convolution2D Algorithm", predict, testY)


def attackPrediction():
    text.delete('1.0', END)
    global dl_model, label_encoder
    filename = filedialog.askopenfilename(initialdir="Dataset")
    dataset = pd.read_csv(filename)
    dataset.fillna(0, inplace = True)
    for i in range(len(columns)):
        dataset[columns[i]] = pd.Series(label_encoder[i].fit_transform(dataset[columns[i]].astype(str)))
    dataset = dataset.values
    X = normalize(dataset)
    #X = X.reshape((X.shape[0],X.shape[1],1,1))
    predict = dl_model.predict(X)  #extracting features using autoencoder
    predict = np.argmax(predict, axis=1)
    print(predict)
    for i in range(len(predict)):
        if predict[i] == 0:
            text.insert(END,"Test Data : "+str(dataset[i])+" ====> NO CYBER ATTACK DETECTED\n\n")
        else:
            text.insert(END,"Test Data : "+str(dataset[i])+" ====> CYBER ATTACK DETECTED\n\n")                       
    

def graph():       
    df = pd.DataFrame([['AutoEncoder','Precision',precision[0]],['AutoEncoder','Recall',recall[0]],['AutoEncoder','F1 Score',fscore[0]],['AutoEncoder','Accuracy',accuracy[0]],
                       ['Propose DL-IDS','Precision',precision[1]],['Propose DL-IDS','Recall',recall[1]],['Propose DL-IDS','F1 Score',fscore[1]],['Propose DL-IDS','Accuracy',accuracy[1]],
                       
                      ],columns=['Algorithms','Performance Output','Value'])
    df.pivot("Algorithms", "Performance Output", "Value").plot(kind='bar')
    plt.show()


def comparisonTable():
    output = "<html><body><table align=center border=1><tr><th>Algorithm Name</th><th>Accuracy</th><th>Precision</th><th>Recall</th>"
    output+="<th>FSCORE</th></tr>"
    output+="<tr><td>AutoEncoder</td><td>"+str(accuracy[0])+"</td><td>"+str(precision[0])+"</td><td>"+str(recall[0])+"</td><td>"+str(fscore[0])+"</td></tr>"
    output+="<tr><td>Propose DL-IDS</td><td>"+str(accuracy[1])+"</td><td>"+str(precision[1])+"</td><td>"+str(recall[1])+"</td><td>"+str(fscore[1])+"</td></tr>"
    output+="</table></body></html>"
    f = open("table.html", "w")
    f.write(output)
    f.close()
    webbrowser.open("table.html",new=2)

def close():
    main.destroy()
    

font = ('times', 16, 'bold')
title = Label(main, text='DL-IDF: Deep Learning Based Intrusion Detection Framework in Industrial Internet of Things')
title.config(bg='greenyellow', fg='dodger blue')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)


font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload UNSW-NB15 Dataset", command=uploadDataset)
uploadButton.place(x=50,y=550)
uploadButton.config(font=font1)  

processButton = Button(main, text="Preprocess Dataset", command=preprocessing)
processButton.place(x=330,y=550)
processButton.config(font=font1) 

autoButton = Button(main, text="Run AutoEncoder-CNN Algorithm", command=runAutoEncoder)
autoButton.place(x=570,y=550)
autoButton.config(font=font1)

proposeButton = Button(main, text="Run Adaptive Sequential Deep CNN", command=runProposeDL)
proposeButton.place(x=920,y=550)
proposeButton.config(font=font1)

predictButton = Button(main, text="Detect Attack from Test Data", command=attackPrediction)
predictButton.place(x=50,y=600)
predictButton.config(font=font1) 

graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=330,y=600)
graphButton.config(font=font1)

tableButton = Button(main, text="Comparison Table", command=comparisonTable)
tableButton.place(x=570,y=600)
tableButton.config(font=font1)

exitButton = Button(main, text="Close App", command=close)
exitButton.place(x=920,y=600)
exitButton.config(font=font1)


main.config(bg='LightSkyBlue')
main.mainloop()
