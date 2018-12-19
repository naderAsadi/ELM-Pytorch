import argparse
import torch
import torchvision.datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from models import ELM, to_onehot


parser = argparse.ArgumentParser(description='Defensive GAN')
parser.add_argument('--hsize', type=int, default=500, help='Number of neurons in hidden layer.')
opt = parser.parse_args()


#################
# Parameters
#################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_size = 28*28
hidden_size = opt.hsize
num_classes = 10

##################
# Datasets
##################
transform = transforms.Compose([
    transforms.ToTensor(),
])
dataset = torchvision.datasets.MNIST(root='~/AI/Datasets/mnist/data', train=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='~/AI/Datasets/mnist/data', train=False, transform=transform)

def get_all_data(dataset, num_workers=30, shuffle=False):
    dataset_size = len(dataset)
    data_loader = DataLoader(dataset, batch_size=dataset_size,
                             num_workers=num_workers, shuffle=shuffle)

    for i_batch, sample_batched in enumerate(data_loader):
        images, labels = sample_batched[0].view(len(dataset), -1).to(device), sample_batched[1].to(device)
    return images, labels

train_images , train_labels = get_all_data(dataset, shuffle=True)
train_labels = to_onehot(batch_size=len(dataset), num_classes=num_classes, y=train_labels, device=device)

test_images , test_labels = get_all_data(dataset, shuffle=False)
test_labels = to_onehot(batch_size=len(dataset), num_classes=num_classes, y=test_labels, device=device)



#################
# Model
#################
elm = ELM(input_size=image_size, h_size=hidden_size, num_classes=num_classes, device=device)
elm.fit(train_images, train_labels)
accuracy = elm.evaluate(test_images, test_labels)

print('Accuracy: {}'.format(accuracy))




