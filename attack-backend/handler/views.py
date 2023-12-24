from networks.compression import CompressionNetwork
from networks.decompression import DecompressionNetwork
from networks.classification import Classifier
from networks.generator import Generator
from networks.noise_simulation import AdversarialNoiseSimulator,ImageNoiseSimulator,ConditionImageNoiseSimulator,NumberNoiseSimulation,ChineseNoiseSimulation
import torch
import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from torch.nn import functional as func

from django.shortcuts import render
import json, os
from django.http import JsonResponse
from django.conf import settings
from django.core.files.storage import FileSystemStorage

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

print("[INFO] Initialization Done")


# Create your views here.
def convert_noisy(request):
    if request.method == 'POST' and request.FILES['file']:
        myfile = request.FILES['file']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        image_path = f"media/{filename}"
        image = cv.imread(image_path)[:,:,0]
        image=  image / 127.5 - 1
        image = torch.tensor(np.array(image)).cuda().to(torch.float32)
        image = image.unsqueeze(0)

        with torch.no_grad():
            image = noise_simulator_chinese_white(image).unsqueeze(0)
            image = generator_c2n(image)
        image = image.squeeze().detach().cpu()
        image = (image + 1) * 127.5
        image = np.array(image).astype(np.uint8)
        filename = f"cn_{filename}"
        cv.imwrite(f"media/{filename}",image)
        
        uploaded_file_url = fs.url(filename)
        
        return JsonResponse({'url':  "http://localhost:8000" + uploaded_file_url})
    return JsonResponse({'error': 'Failed to upload file'}, status=400)

def convert(request):
    if request.method == 'POST' and request.FILES['file']:
        myfile = request.FILES['file']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        image_path = f"media/{filename}"
        image = cv.imread(image_path)[:,:,0]
        image=  image / 127.5 - 1
        image = torch.tensor(np.array(image)).cuda().to(torch.float32)
        image = image.unsqueeze(0)

        with torch.no_grad():
            image = generator_c2n(image)
        image = image.squeeze().detach().cpu()
        image = (image + 1) * 127.5
        image = np.array(image).astype(np.uint8)
        filename = f"cn_{filename}"
        cv.imwrite(f"media/{filename}",image)
        
        uploaded_file_url = fs.url(filename)
        
        return JsonResponse({'url':  "http://localhost:8000" + uploaded_file_url})
    return JsonResponse({'error': 'Failed to upload file'}, status=400)

def compress(image,noise,condition):
    if noise=="black_image_class":
        image = noise_simulator_image_black(image).unsqueeze(0)
    elif noise == "black_image_class_condition":
        image = noise_simulator_image_black_condition(image,condition).unsqueeze(0)
    elif noise == "while_image_class_conition":
        image = noise_simulator_number_white_condition(image,condition).unsqueeze(0)
    elif noise == "while_image_class":
        image = noise_simulator_number_white(image).unsqueeze(0)
    return compression(image)

def transmit(latent,noise,condition):
    if noise=="black_latent_class":
        latent = noise_simulator_latent_black_class(latent)
    elif noise=="black_latent_mse":
        latent = noise_simulator_latent_black_mse(latent)
    elif noise=="while_latent_class":
        latent = noise_simulator_latent_white(latent)
    return latent

def decompress(latent):
    return decompression(latent)

def function(image):
    return classifier(image)

def recognize(request):
    if request.method == 'POST':
            request = json.loads(str(request.body)[2:-1])
            noise = request["noise"]
            image_path = request["path"]
            condition = request["condition"]
            condition = torch.tensor(np.array([int(condition)])).to(int)
            condition = func.one_hot(condition,num_classes = 10)
            condition = condition.cuda().to(torch.float32)
            image_path = image_path[image_path.rfind("/")+1:]
            filename = image_path
            image_path = f"media/{filename}"
            image = cv.imread(image_path)[:,:,0]
            image=  image / 127.5 - 1
            image = torch.tensor(np.array(image)).cuda().to(torch.float32)
            image = image.unsqueeze(0)

            image = decompress(transmit(compress(image,noise,condition),noise,condition))
            result = function(image)
            result = torch.max(result,dim = -1)[1][0]
            result = int(result.detach().cpu())

    return JsonResponse({'status':200, 'text':f'{result}'})
