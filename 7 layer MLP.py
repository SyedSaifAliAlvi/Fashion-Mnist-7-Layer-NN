import pandas as pd
import numpy as np
import os
import gzip
import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.nn import Linear,ReLU,CrossEntropyLoss,Sequential,LeakyReLU
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

#For GPU in Google Colab
#cuda0=torch.device('cuda:0')
#torch.cuda.set_device(cuda0)


def load_mnist(path, kind='train'):

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,'%s-labels-idx1-ubyte.gz'% kind)
    images_path = os.path.join(path,'%s-images-idx3-ubyte.gz'% kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,offset=16).reshape(len(labels), 784)

    return images, labels

class DataPreparation(Dataset):

    def __init__(self, X, y):
        if not torch.is_tensor(X):
            self.X = torch.from_numpy(X)
        if not torch.is_tensor(y):
            self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def model_creation(learning_rate):
  model = Sequential(Linear(input_num_units,hidden_num_units1,bias=True),ReLU(),
                   Linear(hidden_num_units1,hidden_num_units2,bias=True),ReLU(),
                   Linear(hidden_num_units2,hidden_num_units3,bias=True),ReLU(),
                   Linear(hidden_num_units3,hidden_num_units4,bias=True),ReLU(),
                   Linear(hidden_num_units4,hidden_num_units5,bias=True),ReLU(),
                   Linear(hidden_num_units5,hidden_num_units6,bias=True),ReLU(),
                   Linear(hidden_num_units6,output_num_units,bias=True))

  loss_fn = CrossEntropyLoss()

  optimizer = Adam(model.parameters(), lr=learning_rate)
  return model,loss_fn,optimizer

input_num_units = 28*28
hidden_num_units1 = 535
hidden_num_units2 = 363
hidden_num_units3 = 253
hidden_num_units4 = 143
hidden_num_units5 = 105
hidden_num_units6 = 80
output_num_units = 10

def data_setter(batch,features,labels):
  ds = DataPreparation(X=features,y=labels)
  ds = DataLoader(ds, batch_size=batch, shuffle=True)
  steps=features.shape[0]/batch
  return ds

def for_back(x,y,model,loss_fn,optimizer,validation=False):

  x=Variable(x).float()
  y=Variable(y).float()
  
  if validation==True:
    pred=model(x)
    loss=loss_fn(pred,y.type(torch.LongTensor))

  
  else:
    pred=model(x)
    loss=loss_fn(pred,y.type(torch.LongTensor))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
  
  return loss.item()

def Model_run(num_epochs,X_train,y_train,X_test,y_test,learning_rate,batch,lossHistory=True):
  train_losses=[]
  val_losses=[]
  steps=X_train.shape[0]/batch
  val_batch=int(X_test.shape[0]/steps)
  model,loss_fun,optim=model_creation(learning_rate)
  ds1 = data_setter(batch,X_train,y_train)
  ds2 = data_setter(val_batch,X_test,y_test)

  for epoch in range(num_epochs):
    i=0
    for (x1,y1),(x2,y2) in zip(ds1,ds2):
      tr_los=for_back(x1,y1,model,loss_fun,optim,validation=False)
      train_losses.append(tr_los)
      test_los=for_back(x2,y2,model,loss_fun,optim,validation=True)
      val_losses.append(test_los)
      i+=1
      if((i+1)%5==0 and lossHistory==True):
        print(f'epoch {epoch+1}/{num_epochs},step {i+1}/{steps},Train loss= {tr_los}, Validation loss= {test_los}')

  return train_losses,val_losses,model

def plot_loss(train_los,val_los):
  plt.plot(train_los, label='Training loss')
  plt.plot(val_los,label='Validation loss')
  plt.legend()
  plt.show()

def AccuracyScore(x,y,model,validation=False):
  if(validation==True):
    x, y = Variable(torch.from_numpy(x)), Variable(torch.from_numpy(y), requires_grad=False)
    pred = model(x.float())
    final_pred = np.argmax(pred.data.numpy(), axis=1)
    print("Validation Accuracy =",accuracy_score(y, final_pred))
  else:
    x, y = Variable(torch.from_numpy(x)), Variable(torch.from_numpy(y), requires_grad=False)
    pred = model(x.float())
    final_pred = np.argmax(pred.data.numpy(), axis=1)
    print("Training Accuracy =",accuracy_score(y, final_pred))

def Final_result(num_epochs,X_train,y_train,X_test,y_test,lr_rate,batch_no):
  train_loss,validation_loss,Real_model=Model_run(num_epochs,X_train,y_train,X_test,y_test,lr_rate,batch_no,lossHistory=True)
  plot_loss(train_loss,validation_loss)
  AccuracyScore(X_train,y_train,Real_model)
  AccuracyScore(X_test,y_test,Real_model,validation=True)

X_train, y_train = load_mnist('/content/',kind='train')
X_test, y_test = load_mnist('/content/',kind='t10k')
batch_no=100
num_epochs=12
lr_rate=0.0001
Final_result(num_epochs,X_train,y_train,X_test,y_test,lr_rate,batch_no)
print('For Epochs=',num_epochs,'Learning rate=',lr_rate,'Batch size=',batch_no)
print('________________________________________________________________________________________________________________________')
print('________________________________________________________________________________________________________________________')
