import torch
from torch.optim import Adam
from networks.compression import CompressionNetwork
from networks.decompression import DecompressionNetwork
from networks.noise_simulation import GaussianNoiseSimulator
from networks.classification import Classifier
from datasets.dataset import MNISTDataset
from utils.losses import ReconstructLoss,RegulationLoss,ClassificationLoss
from tqdm import tqdm
from matplotlib import pyplot as plt
import os
import random
import numpy

LEARNING_RATE = 0.001
DATASET_DIR = "datasets"
BATCH_SIZE = 128
ITERS = 1000000
SAVE_DELTA = 10
NOISE_RATE = 0.00001
NOISE_START = -0.1
NOISE_MAX = 0.1
SAVE_DELTA = 500
CLASSIFICATION_WEIGHT = 0.2

compression = CompressionNetwork().cuda().train()
decompression = DecompressionNetwork().cuda().train()
classifier = Classifier().cuda().eval()

if os.path.exists("weights/base/classifier.ckpt"):
    classifier.load_state_dict(torch.load("weights/base/classifier.ckpt"))

noise_simulator = GaussianNoiseSimulator().cuda().eval()

compression_optimizer = Adam(compression.parameters(),LEARNING_RATE)
decompression_optimizer = Adam(decompression.parameters(),LEARNING_RATE)

classification_loss_function = ClassificationLoss()
reconstruction_loss_function = ReconstructLoss()
regulation_function = RegulationLoss()

dataset = MNISTDataset(BATCH_SIZE,DATASET_DIR)

losses = []
progress = tqdm(range(ITERS))
for iteration in progress:
    batch,target = dataset()

    real = batch.squeeze().cuda()
    target = target.cuda()

    compression_optimizer.zero_grad()
    decompression_optimizer.zero_grad()
    latents = compression(real)
    noisy_latents = noise_simulator(latents,random.random()*min(max(0,iteration*NOISE_RATE + NOISE_START),NOISE_MAX))
    pred = decompression(noisy_latents)
    result = classifier(pred)

    loss = reconstruction_loss_function(real,pred) + regulation_function(latents) + CLASSIFICATION_WEIGHT * classification_loss_function(target,result)
    loss.backward()

    compression_optimizer.step()
    decompression_optimizer.step()
    progress.set_postfix({"iter:":iteration,"loss":loss.detach().cpu().mean()})
    losses.append(float(loss.detach().cpu().mean()))

    if iteration % SAVE_DELTA==SAVE_DELTA - 1:
        np_losses = numpy.array(losses)
        x = numpy.arange(np_losses.shape[0])
        plt.plot(x,np_losses)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.show()