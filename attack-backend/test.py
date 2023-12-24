from networks.compression import CompressionNetwork
from networks.decompression import DecompressionNetwork
from networks.classification import Classifier
from networks.generator import Generator
from networks.noise_simulation import AdversarialNoiseSimulator,ImageNoiseSimulator,ConditionImageNoiseSimulator,NumberNoiseSimulation,ChineseNoiseSimulation
import torch
import os
import cv2 as cv
import numpy
from matplotlib import pyplot as plt
from torch.nn import functional as func
TEST_DATA_PATH = "datasets/TEST"
CLASSES = 10

def show(y,x,*image_sets):
    index = 1
    for images in image_sets:
        for i in range(images.shape[0]):
            image = (images[i] + 1) / 2
            image = image.detach().cpu()
            plt.subplot(x,y,index)
            index = index + 1
            plt.imshow(image)
    plt.show()

print("[INFO] Initializing Models")
compression = CompressionNetwork().cuda().eval()
decompression = DecompressionNetwork().cuda().eval()
classifier = Classifier().cuda().eval()
generator_c2n = Generator().cuda().eval()
generator_n2c = Generator().cuda().eval()
classifier = Classifier().cuda().eval()

noise_simulator_latent_white = AdversarialNoiseSimulator().cuda().eval()
noise_simulator_latent_black_mse = AdversarialNoiseSimulator().cuda().eval()
noise_simulator_latent_black_class = AdversarialNoiseSimulator().cuda().eval()
noise_simulator_image_black = ImageNoiseSimulator().cuda().eval()
noise_simulator_image_black_condition = ConditionImageNoiseSimulator().cuda().eval()
noise_simulator_number_white = NumberNoiseSimulation().cuda().eval()
noise_simulator_number_white_condition = ConditionImageNoiseSimulator().cuda().eval()
noise_simulator_chinese_white = ChineseNoiseSimulation().cuda().eval()

if os.path.exists("weights/base/compression.ckpt"):
    compression.load_state_dict(torch.load("weights/base/compression.ckpt"))

if os.path.exists("weights/base/decompression.ckpt"):
    decompression.load_state_dict(torch.load("weights/base/decompression.ckpt"))

if os.path.exists("weights/base/classifier.ckpt"):
    classifier.load_state_dict(torch.load("weights/base/classifier.ckpt"))

if os.path.exists("weights/base/generator_c2n.ckpt"):
    generator_c2n.load_state_dict(torch.load("weights/base/generator_c2n.ckpt"))

if os.path.exists("weights/base/generator_n2c.ckpt"):
    generator_n2c.load_state_dict(torch.load("weights/base/generator_n2c.ckpt"))

if os.path.exists("weights/base/classifier.ckpt"):
    classifier.load_state_dict(torch.load("weights/base/classifier.ckpt"))

if os.path.exists("weights/adversarial_latent_white_class/noise_simulator.ckpt"):
    noise_simulator_latent_white.load_state_dict(torch.load("weights/adversarial_latent_white_class/noise_simulator.ckpt"))

if os.path.exists("weights/adversarial_latent_black_mse/noise_simulator.ckpt"):
    noise_simulator_latent_black_mse.load_state_dict(torch.load("weights/adversarial_latent_black_mse/noise_simulator.ckpt"))

if os.path.exists("weights/adversarial_latent_black_class/noise_simulator.ckpt"):
    noise_simulator_latent_black_class.load_state_dict(torch.load("weights/adversarial_latent_black_class/noise_simulator.ckpt"))

if os.path.exists("weights/adversarial_image_black_class/noise_simulator.ckpt"):
    noise_simulator_image_black.load_state_dict(torch.load("weights/adversarial_image_black_class/noise_simulator.ckpt"))

if os.path.exists("weights/adversarial_image_black_class_condition/noise_simulator.ckpt"):
    noise_simulator_image_black_condition.load_state_dict(torch.load("weights/adversarial_image_black_class_condition/noise_simulator.ckpt"))

if os.path.exists("weights/adversarial_chinese_white_class/chinese_noise_simulator.ckpt"):
    noise_simulator_chinese_white.load_state_dict(torch.load("weights/adversarial_chinese_white_class/chinese_noise_simulator.ckpt"))

if os.path.exists("weights/adversarial_number_white_class_condition/noise_simulator.ckpt"):
    noise_simulator_number_white_condition.load_state_dict(torch.load("weights/adversarial_number_white_class_condition/noise_simulator.ckpt"))

if os.path.exists("weights/adversarial_number_white_class/number_noise_simulator.ckpt"):
    noise_simulator_number_white.load_state_dict(torch.load("weights/adversarial_number_white_class/number_noise_simulator.ckpt"))

print("[INFO] Initializing Test Datas")
c_images = []
n_images = []

for i in range(CLASSES):
    c_file_path = f"{TEST_DATA_PATH}/c_{i}.jpg"
    n_file_path = f"{TEST_DATA_PATH}/n_{i}.jpg"
    c_image = cv.imread(c_file_path)[:,:,0]
    n_image = cv.imread(n_file_path)[:,:,0]
    c_image = c_image / 127.5 - 1
    n_image = n_image / 127.5 - 1
    c_images.append(c_image)
    n_images.append(n_image)

c_reals = torch.tensor(numpy.array(c_images)).cuda().to(torch.float32)
n_reals = torch.tensor(numpy.array(n_images)).cuda().to(torch.float32)

show(CLASSES,2,c_reals,n_reals)

with torch.no_grad():
    print("[INFO] Testing Semantic Conversion")
    n_preds = generator_c2n(c_reals)
    show(CLASSES,2,c_reals,n_preds)

    print("[INFO] Testing Semantic Reconstruction")
    n_latents = compression(n_preds)
    n_reconsturcts = decompression(n_latents)
    show(CLASSES,2,n_preds,n_reconsturcts)

    print("[INFO] Testing Semantic Classification")
    result = classifier(n_reconsturcts)
    result = torch.max(result,dim = -1)[1]
    print(result.tolist())

    print("[INFO] Testing Adversarial Noise at Latent, Train By Class, White Box")
    n_noisy_latents = noise_simulator_latent_white(n_latents)
    delta = n_noisy_latents - n_latents
    delta = float(torch.abs(delta).max()/2)
    n_reconsturcts = decompression(n_noisy_latents)
    result = classifier(n_reconsturcts)
    result = torch.max(result,dim = -1)[1]
    print(result.tolist())
    print(f"Noise Rate:{delta}")
    show(CLASSES,2,n_preds,n_reconsturcts)

    print("[INFO] Testing Adversarial Noise at Chinese Image, Train By Class, White Box")
    c_fakes = noise_simulator_chinese_white(c_reals)
    n_preds = generator_c2n(c_fakes)
    delta = c_fakes - c_reals
    delta = float((delta**2).mean())
    n_latents = compression(n_preds)
    n_reconsturcts = decompression(n_latents)
    result = classifier(n_reconsturcts)
    result = torch.max(result,dim = -1)[1]
    print(result.tolist())
    print(f"Noise Rate:{delta}")
    show(CLASSES,4,c_reals,c_fakes,n_preds,n_reconsturcts)

    n_preds = generator_c2n(c_reals)

    print("[INFO] Testing Adversarial Noise at Image, Train By Class, White Box")
    n_noisy_preds = noise_simulator_number_white(n_preds)
    delta = n_noisy_preds - n_preds
    delta = float((delta**2).mean())
    n_latents = compression(n_noisy_preds)
    n_reconsturcts = decompression(n_latents)
    result = classifier(n_reconsturcts)
    result = torch.max(result,dim = -1)[1]
    print(result.tolist())
    print(f"Noise Rate:{delta}")
    show(CLASSES,3,n_preds,n_noisy_preds,n_reconsturcts)

    print("[INFO] Testing Conditional Adversarial Noise at Image, Train By Class, White Box")
    cond = func.one_hot(torch.randint(0,CLASSES,(CLASSES,)),num_classes = CLASSES)
    cond = cond.cuda().to(torch.float32)
    n_noisy_preds = noise_simulator_number_white_condition(n_preds,cond)
    delta = n_noisy_preds - n_preds
    delta = float((delta**2).mean())
    n_latents = compression(n_noisy_preds)
    n_reconsturcts = decompression(n_latents)
    result = classifier(n_reconsturcts)
    result = torch.max(result,dim = -1)[1]
    cond = torch.max(cond,dim = -1)[1]
    print(cond.tolist())
    print(result.tolist())
    print(f"Noise Rate:{delta}")
    show(CLASSES,3,n_preds,n_noisy_preds,n_reconsturcts)

    print("[INFO] Testing Adversarial Noise at Latent, Train By MSE, Black Box")
    n_noisy_latents = noise_simulator_latent_black_mse(n_latents)
    delta = n_noisy_latents - n_latents
    delta = float(torch.abs(delta).max()/2)
    n_reconsturcts = decompression(n_noisy_latents)
    result = classifier(n_reconsturcts)
    result = torch.max(result,dim = -1)[1]
    print(result.tolist())
    print(f"Noise Rate:{delta}")
    show(CLASSES,2,n_preds,n_reconsturcts)

    print("[INFO] Testing Adversarial Noise at Latent, Train By Class, Black Box")
    n_noisy_latents = noise_simulator_latent_black_class(n_latents)
    delta = n_noisy_latents - n_latents
    delta = float(torch.abs(delta).max()/2)
    n_reconsturcts = decompression(n_noisy_latents)
    result = classifier(n_reconsturcts)
    result = torch.max(result,dim = -1)[1]
    print(result.tolist())
    print(f"Noise Rate:{delta}")
    show(CLASSES,2,n_preds,n_reconsturcts)

    print("[INFO] Testing Adversarial Noise at Image, Train By Class, Black Box")
    n_noisy_preds = noise_simulator_image_black(n_preds)
    delta = n_noisy_preds - n_preds
    delta = float((delta**2).mean())
    n_latents = compression(n_noisy_preds)
    n_reconsturcts = decompression(n_latents)
    result = classifier(n_reconsturcts)
    result = torch.max(result,dim = -1)[1]
    print(result.tolist())
    print(f"Noise Rate:{delta}")
    show(CLASSES,3,n_preds,n_noisy_preds,n_reconsturcts)

    print("[INFO] Testing Conditional Adversarial Noise at Image, Train By Class, Black Box")
    cond = func.one_hot(torch.randint(0,CLASSES,(CLASSES,)),num_classes = CLASSES)
    cond = cond.cuda().to(torch.float32)
    n_noisy_preds = noise_simulator_image_black_condition(n_preds,cond)
    delta = n_noisy_preds - n_preds
    delta = float((delta**2).mean())
    n_latents = compression(n_noisy_preds)
    n_reconsturcts = decompression(n_latents)
    result = classifier(n_reconsturcts)
    result = torch.max(result,dim = -1)[1]
    cond = torch.max(cond,dim = -1)[1]
    print(cond.tolist())
    print(result.tolist())
    print(f"Noise Rate:{delta}")
    show(CLASSES,3,n_preds,n_noisy_preds,n_reconsturcts)