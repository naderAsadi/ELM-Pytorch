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

#####################
# Helper Functions
#####################
def to_onehot(batch_size, num_classes, y, device):
    # One hot encoding buffer that you create out of the loop and just keep reusing
    y_onehot = torch.FloatTensor(batch_size, num_classes).to(device)
    #y = y.type(dtype=torch.long)
    y = torch.unsqueeze(y, dim=1)
    # In your for loop
    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1)

    return y_onehot
