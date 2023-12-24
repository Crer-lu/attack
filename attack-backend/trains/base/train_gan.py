import torch
from torch.optim import Adam
from networks.generator import Generator
from networks.discriminator import Discriminator
from datasets.dataset import MNISTDataset,CHMNISTDataset
from utils.losses import ClassificationLoss,GeneratorLoss,ReconstructLoss
from tqdm import tqdm
import numpy
from matplotlib import pyplot as plt
import os
GENERATOR_LEARNING_RATE = 0.0002
DISCRIMINATOR_LEARNING_RATE = 0.0002
DATASET_DIR = "datasets"
BATCH_SIZE = 64
ITERS = 1000000
SAVE_DELTA = 1000
RESIZE = 28
RECONSTRUCTION_WEIGHT = 1000
CLASSES = 10
START = 2000

generator_c2n = Generator().cuda().train()
generator_n2c = Generator().cuda().train()
discriminator_n = Discriminator().cuda().train()
discriminator_c = Discriminator().cuda().train()

if os.path.exists("weights/base/generator_c2n.ckpt"):
    generator_c2n.load_state_dict(torch.load("weights/base/generator_c2n.ckpt"))

if os.path.exists("weights/base/generator_n2c.ckpt"):
    generator_n2c.load_state_dict(torch.load("weights/base/generator_n2c.ckpt"))

if os.path.exists("weights/base/discriminator_n.ckpt"):
    discriminator_n.load_state_dict(torch.load("weights/base/discriminator_n.ckpt"))

if os.path.exists("weights/base/discriminator_c.ckpt"):
    discriminator_c.load_state_dict(torch.load("weights/base/discriminator_c.ckpt"))


generator_c2n_optimizer = Adam(generator_c2n.parameters(),GENERATOR_LEARNING_RATE)
generator_n2c_optimizer = Adam(generator_n2c.parameters(),GENERATOR_LEARNING_RATE)
discriminator_n_optimizer = Adam(discriminator_n.parameters(),DISCRIMINATOR_LEARNING_RATE)
discriminator_c_optimizer = Adam(discriminator_c.parameters(),DISCRIMINATOR_LEARNING_RATE)

cycle_loss_function = ReconstructLoss()
generator_loss_function = GeneratorLoss()
discriminator_loss_function = ClassificationLoss()

dataset_c = CHMNISTDataset(BATCH_SIZE,DATASET_DIR,RESIZE)
dataset_n = MNISTDataset(BATCH_SIZE,DATASET_DIR,RESIZE)

progress = tqdm(range(ITERS))

for iteration in progress:
    if iteration<START:
        continue
    batch_c,label_c = dataset_c()
    batch_n,label_n = dataset_n()

    real_c = batch_c.cuda()
    real_n = batch_n.cuda()

    label_real_c = label_c.cuda()
    label_fake_c = torch.ones_like(label_real_c) * CLASSES
    label_real_n = label_n.cuda()
    label_fake_n = torch.ones_like(label_real_n) * CLASSES

    discriminator_n_optimizer.zero_grad()
    with torch.no_grad():
        pred_n = generator_c2n(real_c)
    pred_n_result = discriminator_n(pred_n)
    real_n_result = discriminator_n(real_n)
    discriminator_n_loss = discriminator_loss_function(label_fake_n,pred_n_result) + discriminator_loss_function(label_real_n,real_n_result)
    discriminator_n_loss.backward()
    discriminator_n_optimizer.step()

    discriminator_c_optimizer.zero_grad()
    with torch.no_grad():
        pred_c = generator_n2c(real_n)
    pred_c_result = discriminator_c(pred_c)
    real_c_result = discriminator_c(real_c)
    discriminator_c_loss = discriminator_loss_function(label_fake_c,pred_c_result) + discriminator_loss_function(label_real_c,real_c_result)
    discriminator_c_loss.backward()
    discriminator_c_optimizer.step()

    generator_c2n_optimizer.zero_grad()
    generator_n2c_optimizer.zero_grad()

    pred_n_from_c = generator_c2n(real_c)
    pred_n_result = discriminator_n(pred_n_from_c)
    pred_c_from_c = generator_n2c(pred_n_from_c)

    pred_c_from_n = generator_n2c(real_n)
    pred_c_result = discriminator_c(pred_c_from_n)
    pred_n_from_n = generator_c2n(pred_c_from_n)
    
    generator_c2n_loss = cycle_loss_function(real_c,pred_c_from_c) + discriminator_loss_function(label_real_c,pred_n_result)
    generator_n2c_loss = cycle_loss_function(real_n,pred_n_from_n) + discriminator_loss_function(label_real_n,pred_c_result)

    generator_loss = generator_c2n_loss + generator_n2c_loss
    generator_loss.backward()

    generator_c2n_optimizer.step()
    generator_n2c_optimizer.step()
    

    discriminator_n_loss = discriminator_n_loss.detach().cpu()
    discriminator_c_loss = discriminator_c_loss.detach().cpu()
    generator_c2n_loss = generator_c2n_loss.detach().cpu()
    generator_n2c_loss = generator_n2c_loss.detach().cpu()

    progress.set_postfix({  "iter:":iteration,
                            "genertor c2n loss":numpy.array(generator_c2n_loss).mean(),
                            "genertor n2c loss":numpy.array(generator_n2c_loss).mean(),
                            "discriminator n loss":numpy.array(discriminator_n_loss).mean(),
                            "discriminator c loss":numpy.array(discriminator_c_loss).mean()})
    
    if iteration % SAVE_DELTA == SAVE_DELTA - 1:
        real_n = (real_n[:4].detach().cpu() + 1) / 2
        real_c = (real_c[:4].detach().cpu() + 1) / 2
        pred_c = (pred_c[:4].detach().cpu() + 1) / 2
        pred_n = (pred_n[:4].detach().cpu() + 1) / 2
        plt.clf()
        for i in range(pred_c.shape[0]):
            plt.subplot(4,pred_c.shape[0],pred_c.shape[0] * 0 + i + 1)
            plt.imshow(pred_c[i])
            plt.subplot(4,pred_c.shape[0],pred_c.shape[0] * 1 + i + 1)
            plt.imshow(real_n[i])
            plt.subplot(4,pred_c.shape[0],pred_c.shape[0] * 2 + i + 1)
            plt.imshow(pred_n[i])
            plt.subplot(4,pred_c.shape[0],pred_c.shape[0] * 3 + i + 1)
            plt.imshow(real_c[i])
        plt.savefig(f"results/gan_{iteration}.png")

        torch.save(generator_c2n.state_dict(),f"weights/base/generator_c2n_{iteration}.ckpt")
        torch.save(generator_n2c.state_dict(),f"weights/base/generator_n2c_{iteration}.ckpt")
        torch.save(discriminator_n.state_dict(),f"weights/base/discriminator_n_{iteration}.ckpt")
        torch.save(discriminator_c.state_dict(),f"weights/base/discriminator_c_{iteration}.ckpt")

        torch.save(generator_c2n.state_dict(),f"weights/base/generator_c2n.ckpt")
        torch.save(generator_n2c.state_dict(),f"weights/base/generator_n2c.ckpt")
        torch.save(discriminator_n.state_dict(),f"weights/base/discriminator_n.ckpt")
        torch.save(discriminator_c.state_dict(),f"weights/base/discriminator_c.ckpt")