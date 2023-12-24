from networks.compression import CompressionNetwork
from networks.decompression import DecompressionNetwork
from networks.noise_simulation import ConditionImageNoiseSimulator
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
WEIGHT_RECONSTRUCT_IMAGE = 0.3
MAX_WEIGHT_RECONSTRUCT_LATENTS = 0.9
WEIGHT_MISGUIDE_RESULT = 1 - WEIGHT_RECONSTRUCT_IMAGE
MIN_WEIGHT_MISGUIDE_RESULT = 1 - MAX_WEIGHT_RECONSTRUCT_LATENTS
WEIGHT_DECAY = 0.00005

compression = CompressionNetwork().cuda().eval()
decompression = DecompressionNetwork().cuda().eval()
classifier = Classifier().cuda().eval()
# latent_classifier = LatentClassifier().cuda().eval()
noise_simulator = ConditionImageNoiseSimulator().cuda().train()


if os.path.exists("weights/base/compression.ckpt"):
    compression.load_state_dict(torch.load("weights/base/compression.ckpt"))

if os.path.exists("weights/base/decompression.ckpt"):
    decompression.load_state_dict(torch.load("weights/base/decompression.ckpt"))

if os.path.exists("weights/base/classifier.ckpt"):
    classifier.load_state_dict(torch.load("weights/base/classifier.ckpt"))

if os.path.exists("weights/adversarial_number_white_class_condition/noise_simulator.ckpt"):
    noise_simulator.load_state_dict(torch.load("weights/adversarial_number_white_class_condition/noise_simulator.ckpt"))


noise_simulator_optimizer = Adam(noise_simulator.parameters(),LEARNING_RATE)
# latent_classifier_optimizer = Adam(latent_classifier.parameters(),LEARNING_RATE)

dataset = MNISTDataset(BATCH_SIZE,DATASET_DIR)
classification_loss_function = ClassificationLoss()
reconstruct_loss_function = ReconstructLoss()

progress = tqdm(range(ITERS))
for iteration in progress:
    batch,result_real = dataset()
    result_real=result_real.cuda()

    target_cond = func.one_hot(torch.randint(0,CLASSES,(BATCH_SIZE,)),num_classes = CLASSES)
    target_cond = target_cond.cuda().to(torch.float32)

    noise_simulator_optimizer.zero_grad()
    real = batch.squeeze().cuda()
    fake = noise_simulator(real,target_cond)
    latent_real = compression(real)
    latent_fake = compression(fake)
    
    pred_fake=decompression(latent_fake)
    result_fake=classifier(pred_fake)
    # result_real = latent_classifier(latent_real)
    # result_fake = latent_classifier(latent_fake)
    delta_loss = WEIGHT_RECONSTRUCT_IMAGE * reconstruct_loss_function(real,fake)
    label_loss = WEIGHT_MISGUIDE_RESULT * (
        +classification_loss_function(target_cond,result_fake)
        -classification_loss_function(result_real,result_fake))
    noise_loss = delta_loss + label_loss
    noise_loss.backward()
    noise_simulator_optimizer.step()

    # with torch.no_grad():
    #     target_real = classifier(decompression(latent_real))
    #     target_fake = classifier(decompression(latent_fake))
    # latent_real = latent_real.detach()
    # latent_fake = latent_fake.detach()

    # latent_classifier_optimizer.zero_grad()
    # result_real = latent_classifier(latent_real)
    # result_fake = latent_classifier(latent_fake)
    # classifier_real_loss = classification_loss_function(target_real,result_real)
    # classifier_fake_loss = classification_loss_function(target_fake,result_fake)
    # classifier_loss = classifier_real_loss + classifier_fake_loss
    # classifier_loss.backward()
    # latent_classifier_optimizer.step()

    WEIGHT_RECONSTRUCT_IMAGE = min(WEIGHT_RECONSTRUCT_IMAGE + WEIGHT_DECAY,MAX_WEIGHT_RECONSTRUCT_LATENTS)
    WEIGHT_MISGUIDE_RESULT = max(WEIGHT_MISGUIDE_RESULT - WEIGHT_DECAY,MIN_WEIGHT_MISGUIDE_RESULT)

    progress.set_postfix({
        "iter:":iteration,
        # "classifier loss":float(classifier_loss.detach().cpu().mean()),
        "delta loss":float(delta_loss.detach().cpu().mean())/WEIGHT_RECONSTRUCT_IMAGE,
        "label loss":float(label_loss.detach().cpu().mean())/WEIGHT_MISGUIDE_RESULT})
    if iteration % SAVE_DELTA==SAVE_DELTA - 1:
        torch.save(noise_simulator.state_dict(),f"weights/adversarial_number_white_class_condition/noise_simulator_{iteration}.ckpt")
        torch.save(noise_simulator.state_dict(),f"weights/adversarial_number_white_class_condition/noise_simulator.ckpt")
        # torch.save(latent_classifier.state_dict(),f"weights/adversarial_image_black_class_condition/latent_classifier_{iteration}.ckpt")
        # torch.save(latent_classifier.state_dict(),f"weights/adversarial_image_black_class_condition/latent_classifier.ckpt")

        real = real[:4].detach().cpu()
        fake = fake[:4].detach().cpu()
        # result_real = torch.max(result_real[:4],dim = -1)[1].detach().cpu()
        result_real=result_real[:4].detach().cpu()
        result_fake = torch.max(result_fake[:4],dim = -1)[1].detach().cpu()
        # result_fake=result_fake[:4].detach().cpu()
        target_cond = torch.max(target_cond[:4],dim = -1)[1].detach().cpu()
        plt.clf()
        for i in range(real.shape[0]):
            plt.subplot(2,real.shape[0],i + 1)
            plt.imshow(real[i])
            plt.subplot(2,fake.shape[0],fake.shape[0] + i + 1)
            plt.imshow(fake[i])
        plt.savefig(f"results/adversarial_number_white_class_condition_results/{iteration}_({int(result_real[0])}_{int(result_real[1])}_{int(result_real[2])}_{int(result_real[3])})_({int(result_fake[0])}_{int(result_fake[1])}_{int(result_fake[2])}_{int(result_fake[3])})_({int(target_cond[0])}_{int(target_cond[1])}_{int(target_cond[2])}_{int(target_cond[3])}).png")