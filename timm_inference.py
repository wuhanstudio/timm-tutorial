import torch 

import timm
from pprint import pprint

import requests
from PIL import Image
from io import BytesIO

from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.feature_extraction import get_graph_node_names

# model_names = timm.list_models("*vit*", pretrained=True)
# pprint(model_names)

model = timm.create_model('visformer_tiny.in1k', pretrained=True).eval()

# model.default_cfg['classifier']
# model.get_classifier()

# Load an Image
url = 'https://datasets-server.huggingface.co/assets/imagenet-1k/--/default/test/12/image/image.jpg'
image = Image.open(requests.get(url, stream=True).raw)

# Pre-processing
transform = timm.data.create_transform(
    **timm.data.resolve_data_config(model.pretrained_cfg)
)
pprint(model.default_cfg)

image_tensor = transform(image)

# Inference
output = model(image_tensor.unsqueeze(0))

probabilities = torch.nn.functional.softmax(output[0], dim=0)

values, indices = torch.topk(probabilities, 5)

# Print results
IMAGENET_1k_URL = 'https://storage.googleapis.com/bit_models/ilsvrc2012_wordnet_lemmas.txt'
IMAGENET_1k_LABELS = requests.get(IMAGENET_1k_URL).text.strip().split('\n')

pprint([{'label': IMAGENET_1k_LABELS[idx], 'value': val.item()} for val, idx in zip(values, indices)])

# Print model structure
for name, param in model.named_parameters():
    print(name)

# Feature extractor
feature_output = model.forward_features(image_tensor.unsqueeze(0))

feature_extractor = create_feature_extractor(
	model, return_nodes=['stage1.0', 'stage1.1', 'stage1.2', 'stage1.3', 'stage1.4', 'stage1.5', 'stage1.6'])
