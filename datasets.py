from torchvision.datasets import ImageFolder, FGVCAircraft, Flowers102, SVHN, StanfordCars, OxfordIIITPet, Food101
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.transforms import ToTensor
from torch.utils.data import Subset
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from medmnist import INFO
from utils import download_using_axel
import zipfile, tarfile
import medmnist
import os, torch, random
import subprocess

class GPR1200(Dataset):
    def __init__(self, processor, download=True):
        url = "https://visual-computing.com/files/GPR1200/GPR1200.zip"
        root = os.environ.get('DATASET_PATH', 'data')
        folder_name = os.path.join(root, "GPR1200")
        
        if download and not os.path.exists(folder_name):
            download_and_extract_archive(url, download_root=folder_name, extract_root=folder_name, remove_finished=True)
            
        self.image_folder = os.path.join(folder_name, "images")
        images = os.listdir(self.image_folder)
        labels = [int(image.split("_")[0]) for image in images]
        
        self.data = sorted(tuple(zip(images, labels)), key = lambda x : x[1])
        self.processor = processor
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.image_folder, self.data[index][0])
        image = Image.open(image_path).convert("RGB").resize((224 , 224))
        label = self.data[index][1]
        
        if self.processor:
            image = self.processor(images=image, return_tensors="pt")
            image = image['pixel_values'].squeeze()
            
        return image, label
    
class CUB2011Dataset(Dataset):
    """Custom CUB-200-2011 dataset"""
    url = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
    
    def __init__(self, root, split='train', download=True):
        self.root = root
        self.split = split
        self.val_split = 0.1
        
        if download and not os.path.exists(os.path.join(root, 'CUB_200_2011')):
            download_and_extract_archive(self.url, root, extract_root=root)
        
        images_path = os.path.join(root, 'CUB_200_2011', 'images.txt')
        labels_path = os.path.join(root, 'CUB_200_2011', 'image_class_labels.txt')
        split_path = os.path.join(root, 'CUB_200_2011', 'train_test_split.txt')
        
        images_df = pd.read_csv(images_path, sep=' ', names=['img_id', 'filepath'])
        labels_df = pd.read_csv(labels_path, sep=' ', names=['img_id', 'target'])
        split_df = pd.read_csv(split_path, sep=' ', names=['img_id', 'is_train'])
        
        data = images_df.merge(labels_df, on='img_id')
        data = data.merge(split_df, on='img_id')
        
        if split == 'train':
            train_data = data[data['is_train'] == 1].reset_index(drop=True)
            val_size = int(len(train_data) * self.val_split)
            self.data = train_data.iloc[val_size:].reset_index(drop=True)
        elif split == 'val':
            train_data = data[data['is_train'] == 1].reset_index(drop=True)
            val_size = int(len(train_data) * self.val_split)
            self.data = train_data.iloc[:val_size].reset_index(drop=True)
        else:
            self.data = data[data['is_train'] == 0].reset_index(drop=True)

        self.images_dir = os.path.join(root, 'CUB_200_2011', 'images')
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.data.iloc[idx]['filepath'])
        image = Image.open(img_path).convert('RGB')
        label = self.data.iloc[idx]['target'] - 1  # Convert to 0-indexed
        return image, label

class ClassificationDataset(Dataset):
    def __init__(self, dataset_name, split, processor):
        self.dataset_name = dataset_name
        self.split = split
        self.processor = processor
        self.data = self.__download_dataset__()

    def __download_dataset__(self):
        rootpath = os.environ.get('DATASET_PATH', 'data')
        path = os.path.join(rootpath, self.dataset_name)

        if self.dataset_name == "aircraft":
            dataset = FGVCAircraft(root=path, split=self.split, download=True)

        elif self.dataset_name == "flowers102":
            dataset = Flowers102(root=path, split=self.split, download=True)

        elif self.dataset_name == "cub2011":
            dataset = CUB2011Dataset(root=path, split=self.split, download=True)

        elif self.dataset_name == "dogs":
            url = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
            download_using_axel(url, path, "images.tar", 10)
            tar_ref = tarfile.open(os.path.join(path, "images.tar"), 'r')
            tar_ref.extractall(path)
            dataset = ImageFolder(os.path.join(path, "Images"))
            dataset = self._get_split_train_test_val(dataset)
            tar_ref.close()

        elif self.dataset_name == "cars":
            url = "https://github.com/AhmadM-DL/Stanford_Cars_dataset.git"
            subprocess.run(["git", "clone", url, os.path.join(path, "Stanford_Cars_dataset")])
            dataset_train = ImageFolder(os.path.join(path, "Stanford_Cars_dataset", "train"))
            dataset_test = ImageFolder(os.path.join(path, "Stanford_Cars_dataset", "test"))
            if self.split in ["train", "val"]:
                dataset = dataset_train
                dataset = self._get_split_train_val(dataset)
            elif self.split == "test":
                dataset = dataset_test
            else:
                raise ValueError(f"Invalid split: {self.split}")

        elif self.dataset_name == "pets":
            if self.split in ["train", "val"]:
                dataset = OxfordIIITPet(root=path, split="trainval", download=True)
                dataset = self._get_split_train_val(dataset)
            else:
                dataset = OxfordIIITPet(root=path, split="test", download=True)
        
        elif self.dataset_name == "food101":
            if self.split in ["train", "val"]:
                dataset = Food101(root=path, split="train", download=True)
                dataset = self._get_split_train_val(dataset)
            elif self.split == "test":
                dataset = Food101(root=path, split="test", download=True)
            else:
                raise ValueError(f"Invalid split: {self.split}")
            
        elif self.dataset_name == "retinamnist":
            dataclass = INFO[self.dataset_name]['python_class']
            if not os.path.exists(path):  os.mkdir(path)
            dataset = getattr(medmnist, dataclass)(split=self.split, download=True, root=path, as_rgb=True, size=224)
        
        elif self.dataset_name in ["chestmnist", "tissuemnist"]:
            url = f"https://zenodo.org/records/10519652/files/{self.dataset_name}_224.npz?download=1"
            download_using_axel(url, path, f"{self.dataset_name}_224.npz", 10)
            dataclass = INFO[self.dataset_name]['python_class']
            dataset = getattr(medmnist, dataclass)(split= self.split, download=True, root=path, as_rgb=True, size=224)
        
        elif self.dataset_name == "eurosat":
            url = "https://zenodo.org/record/7711810/files/EuroSAT_RGB.zip"
            download_using_axel(url, path, "EuroSAT_RGB.zip", 10)
            zip_ref = zipfile.ZipFile(os.path.join(path, "EuroSAT_RGB.zip"), 'r')
            zip_ref.extractall(path)
            dataset = ImageFolder(os.path.join(path, "EuroSAT_RGB"))
            dataset = self._get_split_train_test_val(dataset)
            zip_ref.close()
        
        elif self.dataset_name == "dtd":
            url = "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"
            download_using_axel(url, path, "dtd-r1.0.1.tar.gz", 10)
            tar_ref = tarfile.open(os.path.join(path, "dtd-r1.0.1.tar.gz"), 'r:gz') 
            tar_ref.extractall(path)
            dataset = ImageFolder(os.path.join(path, "dtd", "images"))
            dataset = self._get_split_train_test_val(dataset)
            tar_ref.close()
        
        elif self.dataset_name == "svhn":
            if self.split in ["train", "val"]:
                dataset = SVHN(root=path, split="train", download=True)
                dataset = self._get_split_train_val(dataset)
            elif self.split == "test":
                dataset = SVHN(root=path, split="test", download=True)
            else:
                raise ValueError(f"Invalid split: {self.split}")
        else:
            raise Exception(f"Dataset {self.dataset_name} is not supported!")
        
        return dataset
    
    def is_multilabel(self):
        return self.dataset_name in ["chestmnist"]
    
    def num_labels(self):
        labels_map = {
            "chestmnist": 14,
            "retinamnist": 5,
            "tissuemnist": 8,
            "aircraft": 100,
            "flowers102": 102,
            "cub2011": 200,
            "eurosat": 10,
            "dtd": 47,
            "svhn": 10,
            "dogs": 120,
            "cars": 196,
            "food101": 101,
            "pets": 37
        }
        if self.dataset_name in labels_map:
            return labels_map[self.dataset_name]
        else:
            raise Exception(f"Dataset {self.dataset_name} is not supported!")
        
    def _get_split_train_test_val(self, dataset, train_ratio=0.8, val_ratio=0.1):
        indices = list(range(len(dataset)))
        random.seed(42)
        random.shuffle(indices)
        train_size = int(train_ratio*len(dataset))
        val_size = int(val_ratio*len(dataset))
        if self.split == "train":
            return Subset(dataset, indices[:train_size])
        elif self.split == 'val':
            return Subset(dataset, indices[train_size:train_size + val_size])
        elif self.split == 'test':
            return Subset(dataset, indices[train_size + val_size:])
        else:
            raise ValueError(f"Invalid split: {self.split}")

    def _get_split_train_val(self, dataset, train_ratio=0.9):
        indices = list(range(len(dataset)))
        random.seed(42)
        random.shuffle(indices)
        train_size = int(train_ratio*len(dataset))
        if self.split == "train":
            return Subset(dataset, indices[:train_size])
        elif self.split == 'val':
            return Subset(dataset, indices[train_size:])
        else:
            raise ValueError(f"Invalid split: {self.split}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        if self.dataset_name in ["aircraft", "flowers102", "cub2011", "dogs", "cars", "pets", "food101"]:
            image, label = item[0], item[1]
        
        elif self.dataset_name in ["retinamnist", "tissuemnist"]:
            image, label = item[0], item[1]
            label = int(label)
        
        elif self.dataset_name == "chestmnist": # Multilabel Classification
            image, raw_labels = item[0], item[1]
            label = torch.zeros(14)
            for i in raw_labels:
                label[i] = 1.0

        elif self.dataset_name == "eurosat":
            image, label = item[0], item[1]
        
        elif self.dataset_name == "dtd":
            image, label = item[0], item[1]
        
        elif self.dataset_name == "svhn":
            image, label = item[0], item[1]
            
        else:
            raise Exception(f"Dataset {self.dataset_name} is not supported!")
        
        if self.processor:
            image = self.processor(images=image, return_tensors="pt")
            image = image['pixel_values'].squeeze()
            
        return image, label
    
def get_dataset(dataset_name, split="train", processor=None):
    if dataset_name == "gpr1200":
        return GPR1200(processor=processor, download=True)
    return ClassificationDataset(dataset_name, split, processor)

def _mock_processor(images, return_tensors):
    images = ToTensor()(images)
    return {'pixel_values': images.unsqueeze(0)}

def _test_dataset(dataset_name):
    image = Image.new('RGB', (224, 224), color = 'red')
    dataset = get_dataset(dataset_name, 'train', _mock_processor)
    print(f"Dataset size: {len(dataset)}")
    for i in range(3):
        image, label = dataset[i]
        print(f"Image shape: {image.shape}, Label: {label}")