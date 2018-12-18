import h5py
import numpy as np
import torch
import torch.nn as nn

###############
# ELM
###############
class ELM():
    def __init__(self, input_size, h_size, num_classes, device=None):
        self._input_size = input_size
        self._h_size = h_size
        self._output_size = num_classes
        self._device = device

        self._alpha = nn.init.uniform_(torch.empty(self._input_size, self._h_size, device=self._device), a=-1., b=1.)
        self._beta = nn.init.uniform_(torch.empty(self._h_size, self._output_size, device=self._device), a=-1., b=1.)

        self._bias = torch.zeros(self._h_size, device=self._device)

        self._activation = torch.sigmoid

    def predict(self, x):
        h = self._activation(torch.add(x.mm(self._alpha), self._bias))
        out = h.mm(self._beta)

        return out

    def fit(self, x, t):
        temp = x.mm(self._alpha)
        H = self._activation(torch.add(temp, self._bias))

        H_pinv = torch.pinverse(H)
        self._beta = H_pinv.mm(t)


    def evaluate(self, x, t):
        y_pred = self.predict(x)
        acc = torch.sum(torch.argmax(y_pred, dim=1) == torch.argmax(t, dim=1)).item() / len(t)
        return acc


###################
# Autoencoder ELM
###################
class AEELM():
    def __init__(self, input_size, h1_size, h2_size, h3_size, num_classes, device=None,
                 alpha1_init=None, alpha2_init=None, alpha3_init=None, beta_init=None):
        self._input_size = input_size
        self._h1_size = h1_size
        self._h2_size = h2_size
        self._h3_size = h3_size
        self._output_size = num_classes
        self._device = device

        if isinstance(alpha1_init, torch.Tensor):
            self._alpha1 = alpha1_init
        else:
            self._alpha1 = nn.init.uniform_(torch.empty(self._input_size, self._h1_size, device=self._device), a=-1.,b=1.)
        if isinstance(alpha2_init, torch.Tensor):
            self._alpha2 = alpha2_init
        else:
            self._alpha2 = nn.init.uniform_(torch.empty(self._h1_size, self._h2_size, device=self._device), a=-1., b=1.)
        if isinstance(alpha3_init, torch.Tensor):
            self._alpha3 = alpha3_init
        else:
            self._alpha3 = nn.init.uniform_(torch.empty(self._h2_size, self._h3_size, device=self._device), a=-1., b=1.)
        if isinstance(beta_init, torch.Tensor):
            self._beta = beta_init
        else:
            self._beta = nn.init.uniform_(torch.empty(self._h3_size, self._output_size, device=self._device), a=-1.,b=1.)

        self._bias1 = torch.zeros(self._h1_size, device=device)
        self._bias2 = torch.zeros(self._h2_size, device=device)
        self._bias3 = torch.zeros(self._h3_size, device=device)
        #self._bias1 = nn.init.uniform_(torch.empty(self._h1_size, device=self._device), a=-1., b=1.)
        #self._bias2 = nn.init.uniform_(torch.empty(self._h2_size, device=self._device), a=-1., b=1.)
        #self._bias3 = nn.init.uniform_(torch.empty(self._h3_size, device=self._device), a=-1., b=1.)

        self._activation = torch.sigmoid

    def predict(self, x):
        h1 = self._activation(torch.add(x.mm(self._alpha1), self._bias1))
        h2 = self._activation(torch.add(h1.mm(self._alpha2), self._bias2))
        h3 = self._activation(torch.add(h2.mm(self._alpha3), self._bias3))
        out = h3.mm(self._beta)

        return out

    def fit(self, x, t):
        self._alpha1 = aeelm(x, input_size=self._input_size, hidden_size=self._h1_size, device=self._device)
        h1 = self._activation(torch.add(x.mm(self._alpha1), self._bias1))

        self._alpha2 = aeelm(h1, input_size=self._h1_size, hidden_size=self._h2_size, device=self._device)
        h2 = self._activation(torch.add(h1.mm(self._alpha2), self._bias2))

        self._alpha3 = aeelm(h2, input_size=self._h2_size, hidden_size=self._h3_size, device=self._device)
        H = self._activation(torch.add(h2.mm(self._alpha3), self._bias3))

        H_pinv = torch.pinverse(H)
        self._beta = H_pinv.mm(t)


    def evaluate(self, x, t):
        y_pred = self.predict(x)
        acc = torch.sum(torch.argmax(y_pred, dim=1) == torch.argmax(t, dim=1)).item() / len(t)
        print('pred: {} - label:{}'.format(torch.argmax(y_pred, dim=1),torch.argmax(t, dim=1) ))
        return acc,torch.argmax(y_pred, dim=1)

    def save(self, filepath):
        with h5py.File(filepath, 'w') as f:
            arc = f.create_dataset('architecture',
                                   data=[self._input_size, self._h1_size, self._h2_size,
                                                  self._h3_size, self._output_size])
            arc.attrs['activation'] = 'sigmoid'.encode('utf-8')
            arc.attrs['loss'] = 'mean_square'.encode('utf-8')
            f.create_group('weights')
            f.create_dataset('weights/alpha1', data=self._alpha1)
            f.create_dataset('weights/alpha2', data=self._alpha2)
            f.create_dataset('weights/alpha3', data=self._alpha3)
            f.create_dataset('weights/beta', data=self._beta)
            f.create_dataset('weights/bias1', data=self._bias1)
            f.create_dataset('weights/bias2', data=self._bias2)
            f.create_dataset('weights/bias3', data=self._bias3)

#####################
# Helper Functions
#####################
def aeelm(x, input_size, hidden_size, device=None):
    alpha = nn.init.uniform_(torch.empty(input_size, hidden_size, device=device), a=-1., b=1.)
    bias = torch.zeros(hidden_size, device=device)

    temp = torch.add(x.mm(alpha), bias)
    H = torch.sigmoid(temp)
    H_pinv = torch.pinverse(H)
    return torch.transpose(H_pinv.mm(x), dim0=0, dim1=1)


def load_model(filepath, device):
    with h5py.File(filepath, 'r') as f:
        alpha1_init = torch.tensor(f['weights/alpha1'][...], device=device)
        alpha2_init = torch.tensor(f['weights/alpha2'][...], device=device)
        alpha3_init = torch.tensor(f['weights/alpha3'][...], device=device)
        beta_init = torch.tensor(f['weights/beta'][...], device=device)
        #bias1_init = f['weights/bias1'][...]
        #bias2_init = f['weights/bias2'][...]
        #bias3_init = f['weights/bias3'][...]
        arc = f['architecture']
        input_size = arc[0]
        h1_size = arc[1]
        h2_size = arc[2]
        h3_size = arc[3]
        output_size = arc[4]
        print(alpha1_init.shape)
        model = AEELM(input_size=input_size, h1_size=h1_size, h2_size=h2_size, h3_size=h3_size, num_classes=output_size,
                    alpha1_init=alpha1_init, alpha2_init=alpha2_init, alpha3_init=alpha3_init,  beta_init=beta_init, device=device)
    return model

def to_onehot(batch_size, num_classes, y, device):
    # One hot encoding buffer that you create out of the loop and just keep reusing
    y_onehot = torch.FloatTensor(batch_size, num_classes).to(device)
    #y = y.type(dtype=torch.long)
    y = torch.unsqueeze(y, dim=1)
    # In your for loop
    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1)

    return y_onehot
