import timm

import requests
from PIL import Image
import matplotlib.pyplot as plt

from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

# Load an Image
url = 'https://datasets-server.huggingface.co/assets/imagenet-1k/--/default/test/12/image/image.jpg'
image = Image.open(requests.get(url, stream=True).raw)

plt.imshow(image)
plt.show()

# Method 1: Only output features
model = timm.create_model('tf_efficientnetv2_b2.in1k', pretrained=True, features_only=True)

print(model.feature_info.module_name())
print(model.feature_info.channels())

# Pre-processing
transform = timm.data.create_transform(
    **timm.data.resolve_data_config(model.pretrained_cfg)
)

image_tensor = transform(image)

# Inference
out = model(image_tensor.unsqueeze(0))

for o in out:
    plt.imshow(o[0].sum(0).detach().numpy())
    plt.show()


# Method 2: Feature Extractor
model = timm.create_model('resnet50', pretrained=True, exportable=True)

# Get node names
nodes, _ = get_graph_node_names(model)
print(nodes)

feature_extractor = create_feature_extractor(
	model, return_nodes=['layer1', 'layer2', 'layer3', 'layer4'])

# Pre-processing
transform = timm.data.create_transform(
    **timm.data.resolve_data_config(model.pretrained_cfg)
)

image_tensor = transform(image)

out = feature_extractor(image_tensor.unsqueeze(0))

for o in out:
    plt.imshow(out[o][0].sum(0).detach().numpy())
    plt.show()


# Method 3: Last layer only
# Feature extractor (Last layer only)
# feature_output = model.forward_features(image_tensor.unsqueeze(0))
# plt.imshow(feature_output[0].sum(0).detach().numpy())
# plt.show()
