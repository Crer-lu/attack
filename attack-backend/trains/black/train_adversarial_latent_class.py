from networks.compression import CompressionNetwork
from networks.decompression import DecompressionNetwork
from networks.noise_simulation import AdversarialNoiseSimulator
from networks.classification import Classifier
from networks.classification import LatentClassifier
from torch.optim import Adam
import torch
import os
from matplotlib import pyplot as plt
from datasets.dataset import MNISTDataset
from tqdm import tqdm
from utils.losses import ClassificationLoss,DeltaLoss
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
WEIGHT_RECONSTRUCT_LATENT = 0.5
MAX_WEIGHT_RECONSTRUCT_LATENTS = 0.92
WEIGHT_MISGUIDE_RESULT = 1 - WEIGHT_RECONSTRUCT_LATENT
MIN_WEIGHT_MISGUIDE_RESULT = 1 - MAX_WEIGHT_RECONSTRUCT_LATENTS
WEIGHT_DECAY = 0.00005

compression = CompressionNetwork().cuda().eval()
decompression = DecompressionNetwork().cuda().eval()
classifier = Classifier().cuda().eval()
noise_simulator = AdversarialNoiseSimulator().cuda().train()
latent_classifier = LatentClassifier().cuda().train()

if os.path.exists("weights/base/compression.ckpt"):
    compression.load_state_dict(torch.load("weights/base/compression.ckpt"))

if os.path.exists("weights/base/decompression.ckpt"):
    decompression.load_state_dict(torch.load("weights/base/decompression.ckpt"))

if os.path.exists("weights/base/classifier.ckpt"):
    classifier.load_state_dict(torch.load("weights/base/classifier.ckpt"))

if os.path.exists("weights/adversarial_latent_black_class/noise_simulator.ckpt"):
    noise_simulator.load_state_dict(torch.load("weights/adversarial_latent_black_class/noise_simulator.ckpt"))

if os.path.exists("weights/adversarial_latent_black_class/latent_classifier.ckpt"):
    latent_classifier.load_state_dict(torch.load("weights/adversarial_latent_black_class/latent_classifier.ckpt"))

noise_simulator_optimizer = Adam(noise_simulator.parameters(),LEARNING_RATE)
latent_classifier_optimizer = Adam(latent_classifier.parameters(),LEARNING_RATE)

dataset = MNISTDataset(BATCH_SIZE,DATASET_DIR)
classification_loss_function = ClassificationLoss()
delta_loss_function = DeltaLoss()

progress = tqdm(range(ITERS))
for iteration in progress:
    batch,_ = dataset()

    real = batch.squeeze().cuda()
    noise_simulator_optimizer.zero_grad()
    latent_real = compression(real)
    latent_fake = noise_simulator(latent_real)
    result_real = latent_classifier(latent_real)
    result_fake = latent_classifier(latent_fake)
    delta_loss = WEIGHT_RECONSTRUCT_LATENT * delta_loss_function(latent_real,latent_fake)
    label_loss = -WEIGHT_MISGUIDE_RESULT * classification_loss_function(result_real,result_fake)
    noise_loss = delta_loss + label_loss
    noise_loss.backward()
    noise_simulator_optimizer.step()

    with torch.no_grad():
        real = decompression(latent_real)
        fake = decompression(latent_fake)
        target_real = classifier(real)
        target_fake = classifier(fake)
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

    WEIGHT_RECONSTRUCT_LATENT = min(WEIGHT_RECONSTRUCT_LATENT + WEIGHT_DECAY,MAX_WEIGHT_RECONSTRUCT_LATENTS)
    WEIGHT_MISGUIDE_RESULT = max(WEIGHT_MISGUIDE_RESULT - WEIGHT_DECAY,MIN_WEIGHT_MISGUIDE_RESULT)

    progress.set_postfix({
        "iter:":iteration,
        "classifier loss":float(classifier_loss.detach().cpu().mean()),
        "delta loss":float(delta_loss.detach().cpu().mean())/WEIGHT_RECONSTRUCT_LATENT,
        "label loss":float(label_loss.detach().cpu().mean())/WEIGHT_MISGUIDE_RESULT})
    if iteration % SAVE_DELTA==SAVE_DELTA - 1:
        torch.save(noise_simulator.state_dict(),f"weights/adversarial_latent_black_class/noise_simulator_{iteration}.ckpt")
        torch.save(noise_simulator.state_dict(),f"weights/adversarial_latent_black_class/noise_simulator.ckpt")
        torch.save(latent_classifier.state_dict(),f"weights/adversarial_latent_black_class/latent_classifier_{iteration}.ckpt")
        torch.save(latent_classifier.state_dict(),f"weights/adversarial_latent_black_class/latent_classifier.ckpt")

        real = real[:4].detach().cpu()
        fake = fake[:4].detach().cpu()
        target_real = torch.max(target_real[:4],dim = -1)[1].detach().cpu()
        target_fake = torch.max(target_fake[:4],dim = -1)[1].detach().cpu()
        plt.clf()
        for i in range(real.shape[0]):
            plt.subplot(2,real.shape[0],i + 1)
            plt.imshow(real[i])
            plt.subplot(2,fake.shape[0],fake.shape[0] + i + 1)
            plt.imshow(fake[i])
        plt.savefig(f"tests/{iteration}_({int(target_real[0])}_{int(target_real[1])}_{int(target_real[2])}_{int(target_real[3])})_({int(target_fake[0])}_{int(target_fake[1])}_{int(target_fake[2])}_{int(target_fake[3])}).png")