from torchvision.datasets import MNIST
from torchvision.transforms import Compose,ToTensor,Normalize,Resize
from torch.utils.data import DataLoader
import numpy
import os
import cv2 as cv
import torch

class MNISTDataset:
    def __init__(self,batch_size,root = "./",size = 28):
        self.batch_size = batch_size
        self.transform_function = Compose([
            Resize((size,size)),
            ToTensor()
        ])

        self.batch_size = batch_size
        self.size = size

        self.train_dataset = MNIST(root = root,train = True,transform = self.transform_function)
        self.train_data_loader = DataLoader(self.train_dataset,batch_size = self.batch_size,shuffle = True,drop_last=True)

        self.test_dataset = MNIST(root = root,train = False,transform = self.transform_function)
        self.test_data_loader = DataLoader(self.test_dataset,batch_size = self.batch_size,shuffle = True,drop_last=True)
    
    def __call__(self,batch_size = None,train = True,size = None):
        if train:
            images = self.train_dataset.data
            labels = self.train_dataset.targets
        else:
            images = self.test_dataset.data
            labels = self.test_dataset.targets
        if batch_size is None:
            batch_size = self.batch_size
        if size is None:
            size = images.shape[0]
        indexes = numpy.random.choice(size,batch_size,replace=False)
        images_array = images[indexes]/127.5 - 1
        labels = labels[indexes].to(int)
        images = []
        for i in range(images_array.shape[0]):
            image = numpy.array(images_array[i])
            image = cv.resize(image,(self.size,self.size))
            images.append(image)
        images = torch.tensor(numpy.array(images)).to(torch.float32)
        
        return images,labels
        
class CHMNISTDataset:
    def __init__(self,batch_size,root = ".",size = 28):
        data_folder = f"{root}/CHMNIST"
        image_folder = f"{data_folder}/data/data"
        image_names = os.listdir(image_folder)
        labels = []
        images = []
        for image_name in image_names:
            name = image_name[:-4]
            sections = name.split("_")
            num = int(sections[-1]) - 1
            if num>9:
                continue
            labels.append(num)
            image=  cv.imread(f"{image_folder}/{image_name}")
            image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
            image = numpy.where(image>20,numpy.ones_like(image),numpy.zeros_like(image))/255
            # image = cv.dilate(image, numpy.ones((3,3)), iterations = 1)
            image = cv.GaussianBlur(image,(5,5),0)

            non_zero = image > 0
            x = numpy.arange(image.shape[0])
            y = numpy.arange(image.shape[1])
            x,y = numpy.meshgrid(x,y)
            x = x[non_zero]
            y = y[non_zero]
            image = image[max(y.min()-5,0):min(y.max() + 5,image.shape[0]),max(x.min()-5,0):min(x.max() + 5,image.shape[1])]
            max_shape = max(image.shape[0],image.shape[1])
            new_image = numpy.zeros((max_shape,max_shape),dtype=image.dtype)
            new_image[(max_shape - image.shape[0])//2:(max_shape + image.shape[0])//2,(max_shape - image.shape[1])//2:(max_shape + image.shape[1])//2] = image
            image = new_image
            
            image = cv.resize(image,(size,size))
            
            image = (image - image.min())/(image.max()-image.min())
            image = image * 2 - 1
            images.append(image)
        self.labels = torch.tensor(numpy.array(labels)).to(torch.float32)
        self.images = torch.tensor(numpy.array(images)).to(torch.float32)
        self.batch_size = batch_size
        self.size = size
    
    def __call__(self,batch_size = None,train = True,size = None):
        if batch_size is None:
            batch_size = self.batch_size
        if size is None:
            size = self.images.shape[0]
        indexes = numpy.random.choice(size,batch_size,replace=False)
        images = self.images[indexes]
        labels = self.labels[indexes].to(int)
        return images,labels


