from networks.compression import CompressionNetwork
from networks.decompression import DecompressionNetwork
from networks.noise_simulation import ImageNoiseSimulator
from networks.classification import Classifier
from networks.classification import LatentClassifier
from torch.optim import Adam
import torch
import os
from matplotlib import pyplot as plt
from datasets.dataset import MNISTDataset
from tqdm import tqdm
from utils.losses import ClassificationLoss,ReconstructLoss
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
WEIGHT_RECONSTRUCT_IMAGE = 0.1
MAX_WEIGHT_RECONSTRUCT_LATENTS = 0.9
WEIGHT_MISGUIDE_RESULT = 1 - WEIGHT_RECONSTRUCT_IMAGE
MIN_WEIGHT_MISGUIDE_RESULT = 1 - MAX_WEIGHT_RECONSTRUCT_LATENTS
WEIGHT_DECAY = 0.0001

compression = CompressionNetwork().cuda().eval()
decompression = DecompressionNetwork().cuda().eval()
classifier = Classifier().cuda().eval()
noise_simulator = ImageNoiseSimulator().cuda().train()
latent_classifier = LatentClassifier().cuda().train()

if os.path.exists("weights/base/compression.ckpt"):
    compression.load_state_dict(torch.load("weights/base/compression.ckpt"))

if os.path.exists("weights/base/decompression.ckpt"):
    decompression.load_state_dict(torch.load("weights/base/decompression.ckpt"))

if os.path.exists("weights/base/classifier.ckpt"):
    classifier.load_state_dict(torch.load("weights/base/classifier.ckpt"))

noise_simulator_optimizer = Adam(noise_simulator.parameters(),LEARNING_RATE)
latent_classifier_optimizer = Adam(latent_classifier.parameters(),LEARNING_RATE)

dataset = MNISTDataset(BATCH_SIZE,DATASET_DIR)
classification_loss_function = ClassificationLoss()
reconstruct_loss_function = ReconstructLoss()

progress = tqdm(range(ITERS))
delta_losses = []
label_losses = []
for iteration in progress:
    batch,_ = dataset(size=SUBSET_SIZE)

    real = batch.squeeze().cuda()
    fake = noise_simulator(real)
    latent_real = compression(real)
    latent_fake = compression(fake)

    noise_simulator_optimizer.zero_grad()
    result_real = latent_classifier(latent_real)
    result_fake = latent_classifier(latent_fake)
    delta_loss = WEIGHT_RECONSTRUCT_IMAGE * reconstruct_loss_function(real,fake)
    label_loss = -WEIGHT_MISGUIDE_RESULT * classification_loss_function(result_real,result_fake)
    noise_loss = delta_loss + label_loss
    noise_loss.backward()
    noise_simulator_optimizer.step()

    with torch.no_grad():
        target_real = classifier(decompression(latent_real))
        target_fake = classifier(decompression(latent_fake))
    latent_real = latent_real.detach()
    latent_fake = latent_fake.detach()

    latent_classifier_optimizer.zero_grad()
    result_real = latent_classifier(latent_real)
    result_fake = latent_classifier(latent_fake)
    classifier_real_loss = classification_loss_function(target_real,result_real)
    classifier_fake_loss = classification_loss_function(target_fake,result_fake)
    classifier_loss = classifier_real_loss + classifier_fake_loss
    classifier_loss.backward()
    latent_classifier_optimizer.step()

    WEIGHT_RECONSTRUCT_IMAGE = min(WEIGHT_RECONSTRUCT_IMAGE + WEIGHT_DECAY,MAX_WEIGHT_RECONSTRUCT_LATENTS)
    WEIGHT_MISGUIDE_RESULT = max(WEIGHT_MISGUIDE_RESULT - WEIGHT_DECAY,MIN_WEIGHT_MISGUIDE_RESULT)

    progress.set_postfix({
        "iter:":iteration,
        "classifier loss":float(classifier_loss.detach().cpu().mean()),
        "delta loss":float(delta_loss.detach().cpu().mean())/WEIGHT_RECONSTRUCT_IMAGE,
        "label loss":float(label_loss.detach().cpu().mean())/WEIGHT_MISGUIDE_RESULT})
    delta_losses.append(float(delta_loss.detach().cpu().mean())/WEIGHT_RECONSTRUCT_IMAGE)
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