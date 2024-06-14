# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 23:14:13 2024

@author: gary
"""
from tensorflow import keras
# Numerical Operations
import math
import numpy as np

# Reading/Writing Data
import pandas as pd
import os
import csv

# For Progress Bar
from tqdm import tqdm

# Pytorch
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
def calRecall(dec,gt):
    recall = np.zeros(dec.shape[1])
    N_AP = np.zeros(dec.shape[1]) # actual positive
    N_TP = np.zeros(dec.shape[1]) # true positive
    for j in range(dec.shape[1]):
        N_AP[j] = np.count_nonzero(gt[:,j])
        N_TP[j] = np.count_nonzero(np.multiply(gt[:,j],dec[:,j]))
        recall[j] = N_TP[j] / N_AP[j]
    return recall
def ExactMatchRate(dec,gt):
    emr = 0
    for i in range(dec.shape[0]):
        if np.all(gt[i] == dec[i]):
            emr += 1
    return emr/dec.shape[0]
def ThresholdDecision(pred):
    #pred = np.reshape(pred, (pred.shape[0],pred.shape[1]))
    dec = np.zeros((pred.shape[0],pred.shape[1]))
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            if pred[i,j] > 0.5:
                dec[i,j] = 1
    return dec
def calFalseAlarmRate(dec,gt):
    far = np.zeros(dec.shape[1])
    N_AN = np.zeros(dec.shape[1]) # actual negative
    N_FP = np.zeros(dec.shape[1]) # false positive
    for j in range(dec.shape[1]):
        N_AN[j] = gt.shape[0] - np.count_nonzero(gt[:,j])
        N_FP[j] = np.count_nonzero((gt[:,j]==0) & (dec[:,j]==1))
        far[j] = N_FP[j] / N_AN[j]
    return far
def calPrecision(dec,gt):
    precision = np.zeros(dec.shape[1])
    N_PP = np.zeros(dec.shape[1]) # predict positive
    N_TP = np.zeros(dec.shape[1]) # true positive
    for j in range(dec.shape[1]):
        N_PP[j] = np.count_nonzero(dec[:,j])
        N_TP[j] = np.count_nonzero(np.multiply(gt[:,j],dec[:,j]))
        precision[j] = N_TP[j] / N_PP[j]
    return precision

def calF1(dec,gt):
    F1 = np.zeros(dec.shape[1])
    recall = calRecall(dec,gt)
    prec = calPrecision(dec,gt)
    F1 = 2 / (np.power(recall,-1) + np.power(prec,-1))
    return F1
def same_seed(seed): 
    '''Fixes random number generator seeds for reproducibility.'''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def train_valid_split(data_set, valid_ratio, seed):
    '''Split provided training data into training set and validation set'''
    valid_set_size = int(valid_ratio * len(data_set)) 
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(seed))
    return np.array(train_set), np.array(valid_set)
def predict(test_loader, model, device):
    model.eval() # Set your model to evaluation mode.
    preds = []
    for x in tqdm(test_loader):
        x = x.to(device)                        
        with torch.no_grad():                   
            pred = model(x)                     
            preds.append(pred.detach().cpu())   
    preds = torch.cat(preds, dim=0).numpy()  
    return preds
class Dataset(Dataset):
    '''
    x: Features.
    y: Targets, if none, do prediction.
    '''
    def __init__(self, x, y=None):
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)
class My_Model(nn.Module):
    def __init__(self, input_dim):
        super(My_Model, self).__init__()
        # TODO: modify model's structure, be aware of dimensions. 
        self.layers = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 122),
            nn.ReLU(),
            nn.Linear(122, 60),
            nn.ReLU(),
            nn.Linear(60, 30),
            nn.ReLU(),
            #nn.Linear(30, 10),
            #nn.ReLU(),
            nn.Linear(30, 3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.layers(x)
        #x = x.squeeze(1) # (B, 1) -> (B)
        return x

def trainer(train_loader, valid_loader, model, config, device):
    #pos_weight = torch.ones([5])  # All weights are equal to 1
    #criterion = nn.MSELoss(reduction='mean') # Define your loss function, do not modify this.
    #criterion = torch.nn.BCEWithLogitsLoss()#reduction="mean", #"sum" 
    criterion = torch.nn.BCELoss( )#reduction="mean", #"sum" 
    # Define your optimization algorithm. 
    # TODO: Please check https://pytorch.org/docs/stable/optim.html to get more available algorithms.
    # TODO: L2 regularization (optimizer(weight decay...) or implement by your self).
    #optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9) 
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    #writer = SummaryWriter() # Writer of tensoboard.

    if not os.path.isdir('./models'):
        os.mkdir('./models') # Create directory of saving models.

    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0

    for epoch in range(n_epochs):
        model.train() # Set your model to train mode.
        loss_record = []

        # tqdm is a package to visualize your training progress.
        train_pbar = tqdm(train_loader, position=0, leave=True)

        for x, y in train_pbar:
            optimizer.zero_grad()               # Set gradient to zero.
            x, y = x.to(device), y.to(device)   # Move your data to device. 
            pred = model(x)             
            loss = criterion(pred, y)
            loss.backward()                     # Compute gradient(backpropagation).
            optimizer.step()                    # Update parameters.
            step += 1
            loss_record.append(loss.detach().item())
            
            # Display current epoch number and loss on tqdm progress bar.
            train_pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})

        mean_train_loss = sum(loss_record)/len(loss_record)
       # writer.add_scalar('Loss/train', mean_train_loss, step)

        model.eval() # Set your model to evaluation mode.
        loss_record = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)

            loss_record.append(loss.item())
            
        mean_valid_loss = sum(loss_record)/len(loss_record)
        print(f'Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
        #writer.add_scalar('Loss/valid', mean_valid_loss, step)

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['save_path']) # Save your best model
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else: 
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            return    
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'seed': 70,      # Your seed number, you can pick your lucky number. :)
    'select_all': True,   # Whether to use all features.
    'valid_ratio': 0.1,   # validation_size = train_size * valid_ratio
    'n_epochs': 1000,     # Number of epochs.            
    'batch_size': 350, 
    'learning_rate': 0.0005,              
    'early_stop': 200,    # If model has not improved for this many consecutive epochs, stop training.     
    'save_path': './models/model.ckpt'  # Your model will be saved here.
}
# Set seed for reproducibility
same_seed(config['seed'])
import pandas as pd

n = pd.read_csv("n.csv")
a1 = pd.read_csv("eb.csv")
b1 = pd.read_csv("eu1.csv")
c1 = pd.read_csv("eu2.csv")
mamb1 = pd.read_csv("mBm1.csv")
mamc1 = pd.read_csv("mBm2.csv")
mbmc1 = pd.read_csv("m1m2.csv")
mambmc1 = pd.read_csv("mBm1m2.csv")
a = a1[0:300]
b = b1[0:300]
c = c1[0:300]
mamb = mamb1[0:60]
mamc = mamc1[0:60]
mbmc = mbmc1[0:60]
mambmc = mambmc1[0:60]
train = np.concatenate([n,a,b,c,mamb,mamc,mbmc,mambmc])

label_train = pd.read_csv("label_train.csv")

train_labela = keras.utils.to_categorical(label_train, 3)
train_labelz = train_labela[:,:,0]
t = np.hstack([train,train_labelz])
np.random.shuffle(t)

train_data_label, valid_data_label = train_valid_split(t, config['valid_ratio'], config['seed'])

x_train , y_train= train_data_label[:,0:243] , train_data_label[:,243:246]
x_valid , y_valid= valid_data_label[:,0:243] , valid_data_label[:,243:246]

train_dataset, valid_dataset = Dataset(x_train, y_train), \
                                Dataset(x_valid, y_valid)
                                

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)

model = My_Model(input_dim=x_train.shape[1]).to(device) # put your model and data on the same computation device.
trainer(train_loader, valid_loader, model, config, device)

model = My_Model(input_dim=x_train.shape[1]).to(device)
model.load_state_dict(torch.load(config['save_path']))

#test
tn = pd.read_csv("tn.csv")
a = a1[300:400]
b = b1[300:400]
c = c1[300:400]
mamb = mamb1[60:100]
mamc = mamc1[60:100]
mbmc = mbmc1[60:100]
mambmc = mambmc1[60:100]
test = np.concatenate([tn,a,b,c,mamb,mamc,mbmc,mambmc])
test_dataset = Dataset(test)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

pred = predict(test_loader, model, device) 
preds = ThresholdDecision(pred)

label_test = pd.read_csv("label_test.csv")

label_teata = keras.utils.to_categorical(label_test, 3)
label_teatz = label_teata[:,:,0]
emr = ExactMatchRate(preds, label_teatz) # train_label, x_pred
recall = calRecall(preds, label_teatz)
far = calFalseAlarmRate(preds, label_teatz)
precision = calPrecision(preds, label_teatz)
F1 = calF1(preds, label_teatz)
print('emr',emr)
print('recall',recall)
print('far',far)
print('precision',precision)
print('F1',F1)







