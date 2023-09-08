# TIMM Tutorial

> TIMM (Torch IMage Models) provides SOTA computer vision models.

As the size of deep learning models and datasets grows, it is more common to fine-tune pretrained models than train a model from scratch.

```
# Model selection
# https://github.com/huggingface/pytorch-image-models/blob/main/results/results-imagenet.csv
```

### ResNet50
```
python timm_resnet_train.py
```

### EfficientNet

```
python timm_efficientnet_train.py
```

### Vision Transformer

```
python timm_vit_train.py
```

### VisFormer

```
python timm_visformer_train.py
```

### Full Example

The dataset is available on Kaggle:

https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria

Please download the dataset into the datasets folder and then split the dataset into train, val, and test:

```
$ pip install split-folders

$ python
>> import splitfolders

# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.

>> splitfolders.ratio("datasets/cell_images/", output="datasets/cell_images_py", ratio=(.8, .1, .1), group_prefix=None, move=False)
```

A full binary image classification example:

```
python timm_cell_images_example.py
```
