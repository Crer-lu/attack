import torch
from torch.optim import Adam
from networks.compression import CompressionNetwork
from networks.decompression import DecompressionNetwork
from networks.noise_simulation import GaussianNoiseSimulator
from networks.generator import Generator
from networks.classification import Classifier
from datasets.dataset import CHMNISTDataset
from utils.losses import ClassificationLoss
from tqdm import tqdm
import os
import random

LEARNING_RATE = 0.001
DATASET_DIR = "datasets"
BATCH_SIZE = 128
ITERS = 1000000
SAVE_DELTA = 10
NOISE_RATE = 0.00001
NOISE_START = -0.1
NOISE_MAX = 0.1
SAVE_DELTA = 2000

compression = CompressionNetwork().cuda().eval()
decompression = DecompressionNetwork().cuda().eval()
classifier = Classifier().cuda().train()
generator = Generator().cuda().eval()

if os.path.exists("weights/base/compression.ckpt"):
    compression.load_state_dict(torch.load("weights/base/compression.ckpt"))

if os.path.exists("weights/base/decompression.ckpt"):
    decompression.load_state_dict(torch.load("weights/base/decompression.ckpt"))

if os.path.exists("weights/base/classifier.ckpt"):
    classifier.load_state_dict(torch.load("weights/base/classifier.ckpt"))

if os.path.exists("weights/base/generator_c2n.ckpt"):
    generator.load_state_dict(torch.load("weights/base/generator_c2n.ckpt"))

noise_simulator = GaussianNoiseSimulator().cuda().eval()

classifier_optimizer = Adam(classifier.parameters(),LEARNING_RATE)

loss_function = ClassificationLoss()

dataset = CHMNISTDataset(BATCH_SIZE,DATASET_DIR)

progress = tqdm(range(ITERS))
for iteration in progress:
    batch,target = dataset()
    real = batch.squeeze().cuda()
    real = generator(real)
    target = target.cuda()

    with torch.no_grad():
        latents = compression(real)
        noisy_latents = noise_simulator(latents,random.random()*min(max(0,iteration*NOISE_RATE + NOISE_START),NOISE_MAX))
        pred = decompression(noisy_latents)

    classifier_optimizer.zero_grad()
    result = classifier(pred)
    loss = loss_function(target,result)
    loss.backward()
    classifier_optimizer.step()

    progress.set_postfix({"iter:":iteration,"loss":loss.detach().cpu().mean()})

    if iteration % SAVE_DELTA==SAVE_DELTA - 1:
        torch.save(classifier.state_dict(),f"weights/base/classifier_{iteration}.ckpt")
        torch.save(classifier.state_dict(),f"weights/base/classifier.ckpt")