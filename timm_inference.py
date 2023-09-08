import torch 

import timm
import numpy as np
from pprint import pprint

import requests
from PIL import Image
import matplotlib.pyplot as plt

model = timm.create_model('tf_efficientnetv2_b2.in1k', pretrained=True).eval()

# Load an Image
url = 'https://datasets-server.huggingface.co/assets/imagenet-1k/--/default/test/12/image/image.jpg'
image = Image.open(requests.get(url, stream=True).raw)

plt.imshow(image)
plt.show()

# Pre-processing
transform = timm.data.create_transform(
    **timm.data.resolve_data_config(model.pretrained_cfg)
)

image_tensor = transform(image)

# Inference
output = model(image_tensor.unsqueeze(0))

probabilities = torch.nn.functional.softmax(output[0], dim=0)

values, indices = torch.topk(probabilities, 5)

# Print results
IMAGENET_1k_URL = 'https://storage.googleapis.com/bit_models/ilsvrc2012_wordnet_lemmas.txt'
IMAGENET_1k_LABELS = requests.get(IMAGENET_1k_URL).text.strip().split('\n')

pprint([{'label': IMAGENET_1k_LABELS[idx], 'value': val.item()} for val, idx in zip(values, indices)])
