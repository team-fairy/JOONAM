import Datasets
import Models

input_size = (360, 250)
EPOCHS = 30

model = Model
dataset = OverSamplingMaskDataset
transform = get_augmentation(size = input_size, use_flip = True,
                             use_color_jitter=False, use_gray_scale=False, use_normalize=True)

CONFIG = {
    "model" : EfficientNet_b0,
    "model_name" : 'EfficientNet_b0',
    "nick_name" : 'oversample_size_360250',
    "transform" : transform,
    "dataset" : dataset,
    "learning_rate" : 1e-04,
    "weight_decay" : 2e-02,
    "train_ratio" : 0.3,
    "batch_size" : 32,
    "epoch" : EPOCHS,
    "optimizer" : torch.optim.Adam,
    "loss_function" : torch.nn.CrossEntropyLoss(),
    "input_size" : input_size,
    "load_size" : None

}

Trainer(**CONFIG).train()