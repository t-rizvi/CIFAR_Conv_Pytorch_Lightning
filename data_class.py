import pytorch_lightning as pl
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split



class CIFARDataModule(pl.LightningDataModule):

    def __init__(self, batch_size=4):
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self):
    	mean = [0.485, 0.456, 0.406]
    	std = [0.229, 0.224, 0.225]
    	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    	mytransform = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.ToTensor(), transforms.Normalize(mean, std)])
    	fashion_mnist_train_val = torchvision.datasets.CIFAR10(root='../datasets', train=True,download=True, transform=mytransform)
    	self.fashion_mnist_test = torchvision.datasets.CIFAR10(root='../datasets', train=False, download=True, transform=transform)
    	torch.manual_seed(0)
    	self.train_dataset, self.val_dataset = random_split(fashion_mnist_train_val, [42000, 8000])
    	torch.manual_seed(torch.initial_seed())
        
    #Define the data loaders that can be called from the trainers
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.fashion_mnist_test, batch_size=self.batch_size)
