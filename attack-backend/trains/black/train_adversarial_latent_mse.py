from networks.compression import CompressionNetwork
from networks.decompression import DecompressionNetwork
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
from utils.losses import DeltaLoss,ReconstructLoss
from torch.nn import functional as func

LEARNING_RATE = 0.001
DATASET_DIR = "datasets"
BATCH_SIZE = 128
SUBSET_SIZE = 2048
ITERS = 1000000
SAVE_DELTA = 10
NOISE_RATE = 0.00001
NOISE_START = -0.1
NOISE_MAX = 0.1
SAVE_DELTA = 1000
CLASSES = 10
WEIGHT_RECONSTRUCT_LATENTS = 0.1
MAX_WEIGHT_RECONSTRUCT_LATENTS = 0.93

WEIGHT_MISGUIDE_RESULT = 1 - WEIGHT_RECONSTRUCT_LATENTS
MIN_WEIGHT_MISGUIDE_RESULT = 1 - MAX_WEIGHT_RECONSTRUCT_LATENTS
WEIGHT_DECAY = 0.00001

compression = CompressionNetwork().cuda().eval()
decompression = DecompressionNetwork().cuda().train()
noise_simulator = AdversarialNoiseSimulator().cuda().train()

if os.path.exists("weights/base/compression.ckpt"):
    compression.load_state_dict(torch.load("weights/base/compression.ckpt"))

if os.path.exists("weights/adversarial_latent_black_mse/decompression.ckpt"):
    decompression.load_state_dict(torch.load("weights/adversarial_latent_black_mse/decompression.ckpt"))

if os.path.exists("weights/adversarial_latent_black_mse/noise_simulator.ckpt"):
    noise_simulator.load_state_dict(torch.load("weights/adversarial_latent_black_mse/noise_simulator.ckpt"))

noise_simulator_optimizer = Adam(noise_simulator.parameters(),LEARNING_RATE)
decompression_optimizer = Adam(decompression.parameters(),LEARNING_RATE)

dataset = MNISTDataset(BATCH_SIZE,DATASET_DIR)
delta_loss_function = DeltaLoss()
reconstruct_loss_function = ReconstructLoss()

progress = tqdm(range(ITERS))
for iteration in progress:
    batch,target = dataset()
    real = batch.squeeze().cuda()
    target = target.cuda()
    with torch.no_grad():
        latents = compression(real)

    decompression_optimizer.zero_grad()
    pred = decompression(latents)
    decompression_loss = reconstruct_loss_function(real,pred)
    decompression_loss.backward()
    decompression_optimizer.step()
    
    noise_simulator_optimizer.zero_grad()
    noisy_latents = noise_simulator(latents)
    pred_normal = decompression(latents)
    pred_noisy = decompression(noisy_latents)    
    delta_loss = WEIGHT_RECONSTRUCT_LATENTS * delta_loss_function(latents,noisy_latents)
    label_loss = -WEIGHT_MISGUIDE_RESULT* reconstruct_loss_function(pred_normal,pred_noisy)
    noise_loss = delta_loss + label_loss
    noise_loss.backward()
    noise_simulator_optimizer.step()

    WEIGHT_RECONSTRUCT_LATENTS = min(WEIGHT_RECONSTRUCT_LATENTS + WEIGHT_DECAY,MAX_WEIGHT_RECONSTRUCT_LATENTS)
    WEIGHT_MISGUIDE_RESULT = max(WEIGHT_MISGUIDE_RESULT - WEIGHT_DECAY,MIN_WEIGHT_MISGUIDE_RESULT)

    progress.set_postfix({
        "iter:":iteration,
        "decompress loss":float(decompression_loss.detach().cpu().mean()),
        "delta loss":float(delta_loss.detach().cpu().mean()**0.5)/WEIGHT_RECONSTRUCT_LATENTS,
        "label loss":float(label_loss.detach().cpu().mean())/WEIGHT_MISGUIDE_RESULT})
    if iteration % SAVE_DELTA==SAVE_DELTA - 1:
        torch.save(noise_simulator.state_dict(),f"weights/adversarial_latent_black_mse/noise_simulator_{iteration}.ckpt")
        torch.save(noise_simulator.state_dict(),f"weights/adversarial_latent_black_mse/noise_simulator.ckpt")
        torch.save(decompression.state_dict(),f"weights/adversarial_latent_black_mse/decompression_{iteration}.ckpt")
        torch.save(decompression.state_dict(),f"weights/adversarial_latent_black_mse/decompression.ckpt")
        pred_normal = pred_normal[:4].detach().cpu()
        pred_noisy = pred_noisy[:4].detach().cpu()
        plt.clf()
        for i in range(pred_normal.shape[0]):
            plt.subplot(2,pred_normal.shape[0],i+1)
            plt.imshow(pred_normal[i])
            plt.subplot(2,pred_normal.shape[0],pred_normal.shape[0] + i+1)
            plt.imshow(pred_noisy[i])
        plt.savefig(f"tests/{iteration}.png")