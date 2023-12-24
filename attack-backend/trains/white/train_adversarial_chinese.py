from networks.compression import CompressionNetwork
from networks.decompression import DecompressionNetwork
from networks.classification import Classifier
from networks.noise_simulation import ChineseNoiseSimulation
from networks.generator import Generator
from torch.optim import Adam
import torch
import os
from matplotlib import pyplot as plt
from datasets.dataset import CHMNISTDataset
from tqdm import tqdm
from utils.losses import ClassificationLoss,ReconstructLoss

LEARNING_RATE = 0.001
DATASET_DIR = "datasets"
BATCH_SIZE = 128
SUBSET_SIZE = 2048
ITERS = 1000000
NOISE_RATE = 0.00001
NOISE_START = -0.1
NOISE_MAX = 0.1
SAVE_DELTA = 1000
CLASSES = 10
WEIGHT_RECONSTRUCT_LATENTS = 0.9
MAX_WEIGHT_RECONSTRUCT_LATENTS = 0.95
WEIGHT_MISGUIDE_RESULT = 0.1
MIN_WEIGHT_MISGUIDE_RESULT = 0.05
WEIGHT_DECAY = 0.00005

compression = CompressionNetwork().cuda().eval()
decompression = DecompressionNetwork().cuda().eval()
classifier = Classifier().cuda().eval()
generator_c2n=Generator().cuda().eval()
noise_simulator = ChineseNoiseSimulation().cuda().train()


if os.path.exists("weights/base/generator_c2n.ckpt"):
    generator_c2n.load_state_dict(torch.load("weights/generator_c2n.ckpt"))

if os.path.exists("weights/base/compression.ckpt"):
    compression.load_state_dict(torch.load("weights/compression.ckpt"))

if os.path.exists("weights/base/decompression.ckpt"):
    decompression.load_state_dict(torch.load("weights/decompression.ckpt"))

if os.path.exists("weights/base/classifier.ckpt"):
    classifier.load_state_dict(torch.load("weights/classifier.ckpt"))

if os.path.exists("weights/adversarial_chinese_white_class/chinese_noise_simulator.ckpt"):
    noise_simulator.load_state_dict(torch.load("weights/adversarial_chinese_white_class/chinese_noise_simulator.ckpt"))

noise_simulator_optimizer = Adam(noise_simulator.parameters(),LEARNING_RATE)
dataset = CHMNISTDataset(BATCH_SIZE,DATASET_DIR)
classification_loss_function = ClassificationLoss()
# reconstruct_loss_function = DeltaLoss()
reconstruct_loss_function=ReconstructLoss()

progress = tqdm(range(ITERS))
for iteration in progress:
    batch,target = dataset(size=SUBSET_SIZE)
    real_c = batch.cuda()
    target = target.cuda()
    
    noise_simulator_optimizer.zero_grad()
    noisy_c=noise_simulator(real_c)

    noisy_n=generator_c2n(noisy_c)
    noisy_n_squeeze=noisy_n.squeeze().cuda()
    # with torch.no_grad():
    latents=compression(noisy_n_squeeze)
    pred_n = decompression(latents)
    result = classifier(pred_n)
    delta_loss = WEIGHT_RECONSTRUCT_LATENTS * reconstruct_loss_function(real_c,noisy_c)
    label_loss = - WEIGHT_MISGUIDE_RESULT* classification_loss_function(target,result)
    WEIGHT_RECONSTRUCT_LATENTS = min(WEIGHT_RECONSTRUCT_LATENTS + WEIGHT_DECAY,MAX_WEIGHT_RECONSTRUCT_LATENTS)
    WEIGHT_MISGUIDE_RESULT = max(WEIGHT_MISGUIDE_RESULT - WEIGHT_DECAY,MIN_WEIGHT_MISGUIDE_RESULT)
    loss = delta_loss + label_loss
    loss.backward()
    noise_simulator_optimizer.step()
    progress.set_postfix({
        "iter:":iteration,"delta loss":float(delta_loss.detach().cpu().mean()**0.5)/WEIGHT_RECONSTRUCT_LATENTS,
        "label loss":float(label_loss.detach().cpu().mean())/WEIGHT_MISGUIDE_RESULT})
    if iteration % SAVE_DELTA==SAVE_DELTA - 1:
        # torch.save(noise_simulator.state_dict(),f"weights/number_noise_simulator_{iteration}.ckpt")
        torch.save(noise_simulator.state_dict(),f"weights/adversarial_chinese_white_class/chinese_noise_simulator.ckpt")
        real_c_squeeze = real_c.squeeze()[:4].detach().cpu()
        noisy_c_squeeze = noisy_c.squeeze()[:4].detach().cpu()
        noisy_n_squeeze = noisy_n_squeeze[:4].detach().cpu()
        pred_n = pred_n[:4].detach().cpu()
        target = target[:4].detach().cpu()
        result = result[:4].detach().cpu()
        _,result = torch.max(result,dim = -1)
        plt.clf()
        for i in range(real_c_squeeze.shape[0]):
            plt.subplot(4,real_c_squeeze.shape[0],i+1)
            plt.imshow(real_c_squeeze[i])
            plt.subplot(4,real_c_squeeze.shape[0],real_c_squeeze.shape[0] + i+1)
            plt.imshow(noisy_c_squeeze[i])
            plt.subplot(4,real_c_squeeze.shape[0],real_c_squeeze.shape[0]*2 + i+1)
            plt.imshow(noisy_n_squeeze[i])
            plt.subplot(4,real_c_squeeze.shape[0],real_c_squeeze.shape[0]*3 + i+1)
            plt.imshow(pred_n[i])
        plt.savefig(f"results/adversarial_chinese_white_class_results/{iteration}_({int(target[0])}_{int(target[1])}_{int(target[2])}_{int(target[3])})_({int(result[0])}_{int(result[1])}_{int(result[2])}_{int(result[3])}).png")