from transformers import AutoImageProcessor, AutoModel, ViTImageProcessor, DPTForDepthEstimation
from torch.nn.functional import adaptive_avg_pool1d
from torch import no_grad
import torchvision
import torch
import timm
from torchvision import transforms
from PIL import Image
import numpy as np

def get_encoder(encoder_id, device="cuda"):

    if "custom" in encoder_id.lower():

        if "byol" in encoder_id.lower():
            checkpoint_url = "https://github.com/AhmadM-DL/BYOL-Imagenet1k-Resnet50-weights/raw/refs/heads/main/pretrain_res50x1.pth"
            checkpoint = torch.hub.load_state_dict_from_url(checkpoint_url, progress=True)
            encoder = torchvision.models.resnet50()
            encoder.load_state_dict(checkpoint)
            encoder.fc = torch.nn.Identity()
            encoder.to(device)
            image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50", use_fast=True)
        
        if "moco" in encoder_id.lower():
            checkpoint_url = "https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar"
            checkpoint = torch.hub.load_state_dict_from_url(checkpoint_url, progress=True)
            state_dict = checkpoint["state_dict"]
            # Keep only base encoder(without head) and remove everything else (momentum, predictor)
            for k in list(state_dict.keys()):
                if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.head'):
                    state_dict[k.replace("module.base_encoder.", "")] = state_dict[k]
                del state_dict[k]
            model = timm.create_model('vit_base_patch16_224', pretrained=False)
            # Drop head from ViT model
            model.head= torch.nn.Identity()
            try:
                model.load_state_dict(state_dict)
            except:
                print("Timm model or checkpoint architecture changed")
            model.to(device)
            encoder = model
            image_processor = _CustomImageProcessor()

        if "simclr" in encoder_id.lower():
            checkpoint_url = "https://github.com/AhmadM-DL/SimCLR-ImageNet1k-Resnet50-weights/raw/refs/heads/main/simclr_resnet50_1x_sk0.pth"
            checkpoint = torch.hub.load_state_dict_from_url(checkpoint_url, progress=True)
            encoder = torchvision.models.resnet50()
            encoder.load_state_dict(checkpoint)
            encoder.fc = torch.nn.Identity()
            encoder.to(device)
            image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50", use_fast=True)
        
        if "eva" in encoder_id.lower():
            encoder = timm.create_model('eva02_base_patch14_224.mim_in22k', pretrained=True)
            encoder= encoder.to(device)
            image_processor = _CustomImageProcessor()
        
    elif 'dpt' in encoder_id.lower():
        image_processor = AutoImageProcessor.from_pretrained(encoder_id, use_fast=True)
        encoder = DPTForDepthEstimation.from_pretrained(encoder_id).to(device)   

    else:    
        image_processor = AutoImageProcessor.from_pretrained(encoder_id, use_fast=True)
        encoder = AutoModel.from_pretrained(encoder_id).to(device)

    return encoder, image_processor

def pool_features(features, to_dimensionality):
    batch_size = features.shape[0]
    current_dim = features.shape[1]

    if current_dim < to_dimensionality:
        raise Exception(f"Error: Model output dim {current_dim} is less than target dim {to_dimensionality}")
    
    if current_dim == to_dimensionality:
        return features
    
    pooled_features = adaptive_avg_pool1d(features, to_dimensionality)
    pooled_features = pooled_features.view(batch_size, to_dimensionality)
    return pooled_features

def get_features(encoder, X, target_dim, device="cuda"):
    if not len(X.shape) == 4:
        raise Exception("The function expect a tensor of 4 dimensions.")
    
    X = X.to(device)
    
    with no_grad():
        
        # Clip
        if "clip" in str(type(encoder)):
          outputs = encoder.vision_model(X)
          features = outputs.last_hidden_state[:, 0, :]
          features = pool_features(features, target_dim)

        # Models loaded using timm and torch vision
        elif "timm" in str(type(encoder)) or "torchvision" in str(type(encoder)):
            outputs = encoder(X)
            features = pool_features(outputs, target_dim)
        
        # Convolutional models
        elif "resnet" in str(type(encoder)) or "efficientnet" in str(type(encoder)) or "convnext" in str(type(encoder)):
            outputs = encoder(X)
            features = outputs.pooler_output
            if len(features.shape) == 1:
                features = features.unsqueeze(0)
            elif len(features.shape) > 2:
                features = features.squeeze()
            features = pool_features(features, target_dim)
        
        # Swin and MAE AVG
        elif "swin" in str(type(encoder)) or "mae" in str(type(encoder)):
            outputs = encoder(X, output_hidden_states= True)
            features = outputs.hidden_states[-1]
            features = features.mean(dim=1)
            features = pool_features(features, target_dim)

        # MiDaS AVG
        elif "dpt" in str(type(encoder)):
            outputs = encoder.backbone(X, output_hidden_states= True)
            features = outputs.hidden_states[-1]
            features = features.mean(dim=1)
            features = pool_features(features, target_dim)
                
        # Other transformer models [CLS]
        elif "deit" in str(type(encoder)) or "vit" in str(type(encoder)) or "dino" in str(type(encoder)) or "eva" in str(type(encoder)):
            outputs = encoder(X)
            features = outputs.last_hidden_state
            features = features[:, 0, :]
            features = pool_features(features, target_dim)
        
        else:
            raise Exception(f"The encoder {str(type(encoder))} is not supported!")

    return features

class _CustomImageProcessor:
    def __init__(
        self,
        image_size=224,
        resize_size=256,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        self.image_size = image_size
        self.resize_size = resize_size

        self.transform = transforms.Compose([
            transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __call__(self, images, return_tensors="pt"):
        if isinstance(images, (Image.Image, np.ndarray)):
            images = [images]
        elif isinstance(images, (list, tuple)):
            if not all(isinstance(img, (Image.Image, np.ndarray)) for img in images):
                raise ValueError("All images must be PIL.Image or numpy.ndarray")
            images = list(images)
        else:
            raise ValueError(f"Unsupported input type: {type(images)}")
        processed = []
        for img in images:
            if isinstance(img, np.ndarray):
                if img.ndim == 2:  # grayscale
                    img = Image.fromarray(img, mode="L")
                elif img.ndim == 3:
                    img = Image.fromarray(img)
                else:
                    raise ValueError(f"Invalid ndarray shape: {img.shape}")
            tensor = self.transform(img)
            processed.append(tensor)
        pixel_values = torch.stack(processed)
        if return_tensors == "pt":
            return {"pixel_values": pixel_values}
        raise ValueError(f"Unsupported return_tensors={return_tensors}")

def _test_encoder(encoder_id):
    batch_size = 32
    X = torch.rand((batch_size, 3, 224, 224)).to("cuda")
    encoder, img_processor = get_encoder(encoder_id)
    X = img_processor(X, return_tensors="pt")["pixel_values"]
    target_dim = 768
    features = get_features(encoder, X, target_dim=target_dim)
    assert features.shape == (batch_size, target_dim)