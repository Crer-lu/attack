from networks.compression import CompressionNetwork
from networks.decompression import DecompressionNetwork
from networks.classification import Classifier
from networks.generator import Generator
from networks.noise_simulation import AdversarialNoiseSimulator
from torch.optim import Adam
import torch
import os
import cv2 as cv
import numpy
from matplotlib import pyplot as plt
from datasets.dataset import MNISTDataset
from tqdm import tqdm
from utils.losses import ClassificationLoss,DeltaLoss
from torch.nn import functional as func
import numpy
LEARNING_RATE = 0.001
DATASET_DIR = "datasets"
BATCH_SIZE = 128
SUBSET_SIZE = 2048
ITERS = 1000000
SAVE_DELTA = 10
NOISE_RATE = 0.00001
NOISE_START = -0.1
NOISE_MAX = 0.1
SAVE_DELTA = 5000
CLASSES = 10
WEIGHT_RECONSTRUCT_LATENTS = 0.1
MAX_WEIGHT_RECONSTRUCT_LATENTS = 0.9
WEIGHT_MISGUIDE_RESULT = 1 - WEIGHT_RECONSTRUCT_LATENTS
MIN_WEIGHT_MISGUIDE_RESULT = 1 - MAX_WEIGHT_RECONSTRUCT_LATENTS
WEIGHT_DECAY = 0.0001

compression = CompressionNetwork().cuda().eval()
decompression = DecompressionNetwork().cuda().eval()
classifier = Classifier().cuda().eval()
noise_simulator = AdversarialNoiseSimulator().cuda().train()

if os.path.exists("weights/base/compression.ckpt"):
    compression.load_state_dict(torch.load("weights/base/compression.ckpt"))

if os.path.exists("weights/base/decompression.ckpt"):
    decompression.load_state_dict(torch.load("weights/base/decompression.ckpt"))

if os.path.exists("weights/base/classifier.ckpt"):
    classifier.load_state_dict(torch.load("weights/base/classifier.ckpt"))


noise_simulator_optimizer = Adam(noise_simulator.parameters(),LEARNING_RATE)

dataset = MNISTDataset(BATCH_SIZE,DATASET_DIR)
classification_loss_function = ClassificationLoss()
reconstruct_loss_function = DeltaLoss()

progress = tqdm(range(ITERS))
delta_losses = []
label_losses = []
for iteration in progress:
    batch,target = dataset()
    real = batch.squeeze().cuda()
    target = target.cuda()
    with torch.no_grad():
        latents = compression(real)
    
    noise_simulator_optimizer.zero_grad()

    noisy_latents = noise_simulator(latents)
    pred_normal = decompression(latents)
    pred_noisy = decompression(noisy_latents)
    result = classifier(pred_noisy)
    delta_loss = WEIGHT_RECONSTRUCT_LATENTS * reconstruct_loss_function(latents,noisy_latents)
    label_loss = - WEIGHT_MISGUIDE_RESULT* classification_loss_function(target,result)
    loss = delta_loss + label_loss
    loss.backward()
    noise_simulator_optimizer.step()

    WEIGHT_RECONSTRUCT_LATENTS = min(WEIGHT_RECONSTRUCT_LATENTS + WEIGHT_DECAY,MAX_WEIGHT_RECONSTRUCT_LATENTS)
    WEIGHT_MISGUIDE_RESULT = max(WEIGHT_MISGUIDE_RESULT - WEIGHT_DECAY,MIN_WEIGHT_MISGUIDE_RESULT)

    progress.set_postfix({
        "iter:":iteration,
        "delta loss":float(delta_loss.detach().cpu().mean()**0.5)/WEIGHT_RECONSTRUCT_LATENTS,
        "label loss":float(label_loss.detach().cpu().mean())/WEIGHT_MISGUIDE_RESULT})
    delta_losses.append(float(delta_loss.detach().cpu().mean())/WEIGHT_RECONSTRUCT_LATENTS)
    label_losses.append(float(label_loss.detach().cpu().mean())/WEIGHT_MISGUIDE_RESULT)
    if iteration % SAVE_DELTA==SAVE_DELTA - 1:
        np_delta_losses = numpy.array(delta_losses)
        np_label_losses = numpy.array(label_losses)
        x = numpy.arange(np_delta_losses.shape[0])
        plt.plot(x,delta_losses)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.show()
        plt.plot(x,np_label_losses)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.show()