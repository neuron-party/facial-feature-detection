import torch
import torch.nn as nn
import numpy as np
import pickle


class Model:
    def __init__(self, network, lr=1e-3):
        self.network = network
        self.optim = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.loss_history = {'training':[], 'validation':[]}

    def update(self, *args, **kwargs):
        raise NotImplementedError
    
    def eval(self, *args, **kwargs):
        raise NotImplementedError
    
    def load(self, path, map_location=None):
        checkpoint = torch.load(path, map_location=map_location)
        self.network.load_state_dict(checkpoint['network'])
        self.optim.load_state_dict(checkpoint['optim'])
        self.loss_history = checkpoint['loss_history']

    def save(self, path):
        torch.save({
            'network': self.network.state_dict(),
            'optim': self.optim.state_dict(),
            'loss_history': self.loss_history,
        }, path)


class ModelCAE(Model): #redraws the face
    def __init__(self, network, lr=1e-3):
        super().__init__(network, lr)
        self.loss = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    
    def update(self, x):
        self.optim.zero_grad()
        x_pred = self.network(x)
        loss = self.loss(x_pred, x)
        loss.backward()
        self.optim.step()
        self.loss_history['training'].append(float(loss.cpu().detach()))
    
    def eval(self, x):
        x_pred = self.network(x)
        loss = self.loss(x_pred, x)
        self.loss_history['validation'].append(float(loss.cpu().detach()))
        return loss


class ModelSigmoidClassifier(Model):
    """
    For binary classification and multi-label classification.
    """
    def __init__(self, network, lr=1e-3):
        super().__init__(network, lr)
        self.loss = nn.BCEWithLogitsLoss(weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None)
    
    def update(self, x, y):
        self.optim.zero_grad()
        y_pred = self.network(x)
        loss = self.loss(y_pred, y)
        loss.backward() #calculates weights 
        self.optim.step() #updates weights
        self.loss_history['training'].append(float(loss.cpu().detach()))
    
    def eval(self, x, y):
        y_pred = self.network(x)
        loss = self.loss(y_pred, y)
        self.loss_history['validation'].append(float(loss.cpu().detach()))
        return loss


class ModelSoftmaxClassifier(Model):
    """
    For single-label classification.
    """
    def __init__(self, network, lr=1e-3):
        super().__init__(network, lr)
        self.loss = nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
    
    def update(self, x, y):
        self.optim.zero_grad()
        y_pred = self.network(x)
        loss = self.loss(y_pred, y)
        loss.backward()
        self.optim.step()
        self.loss_history['training'].append(float(loss.cpu().detach()))

    def eval(self, x, y):
        y_pred = self.network(x)
        loss = self.loss(y_pred, y)
        self.loss_history.append(float(loss.cpu().detach()))
        self.loss_history['validation'].append(float(loss.cpu().detach()))
        return loss


class ModelRegressor(Model): #unused
    """
    For regression.
    """
    def __init__(self, network, lr=1e-3):
        super().__init__(network, lr)
        self.loss = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    
    def update(self, x, y):
        self.optim.zero_grad()
        y_pred = self.network(x)
        loss = self.loss(y_pred, y)
        loss.backward()
        self.optim.step()
        self.loss_history['training'].append(float(loss.cpu().detach()))
    
    def eval(self, x, y):
        y_pred = self.network(x)
        loss = self.loss(y_pred, y)
        self.loss_history['validation'].append(float(loss.cpu().detach()))
        return loss