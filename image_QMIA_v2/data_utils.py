import contextlib
import os

from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

from typing import NamedTuple, Optional, Tuple, Any
from PIL import Image, ImageFile

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import torchvision.datasets as tv_datasets

def download_imagenet_folder(
    hf_name: str = "evanarlian/imagenet_1k_resized_256",
    data_root: str = "./data",
):
    data_root = Path(data_root)
    dataset_name = hf_name.split("/")[-1]

    # 1) Grab the class‐name mapping from the train split:
    train_ds = load_dataset(hf_name, split="train", streaming=True)
    class_names = train_ds.features["label"].names

    # 2) For each split, stream and write:
    for split in ("train", "val", "test"):
        ds = load_dataset(hf_name, split=split, streaming=True)
        out_base = data_root / dataset_name / split

        # You can pass total=len(...) if you really want a pbar length.
        pbar = tqdm(ds, desc=f"Saving {split}")
        for i, ex in enumerate(pbar):
            img = ex["image"]        # a PIL.Image
            lbl = ex["label"]        # integer 0…999
            cname = class_names[lbl]
            out_dir = out_base / cname
            out_dir.mkdir(parents=True, exist_ok=True)
            img.save(out_dir / f"{i:08d}.jpeg")

    print("✅ All splits saved under", data_root / dataset_name)

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

class DATASET_FLAGS(NamedTuple):
    DATA_SEED = 42
    MNIST_MEAN = (0.1307,)
    MNIST_STD = (0.3081,)
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD = (0.2023, 0.1994, 0.2010)
    CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
    CIFAR100_STD = (0.2675, 0.2565, 0.2761)
    CINIC10_MEAN = (0.47889522, 0.47227842, 0.43047404)
    CINIC10_STD = (0.24205776, 0.23828046, 0.25874835)
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

def set_transform(dataset, transform):
    """
    Set the transform for a dataset. 
    This is useful for removing transforms from
    datasets for evaluation.
    """
    if hasattr(dataset, "transform"):
        dataset.transform = transform
    elif hasattr(dataset, "dataset"):
        if hasattr(dataset.dataset, "transform"):
            dataset.dataset.transform = transform
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return dataset

class PairedCustomCIFAR100(tv_datasets.CIFAR100):
    def __init__(self, size=-1, base_size=-1, mean=None, std=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if size == -1:
            self.resize_transform = transforms.Lambda(lambda x: x)
        else:
            self.resize_transform = transforms.Compose(
                [
                    transforms.Resize(size),
                ]
            )

        if base_size == -1:
            self.base_resize_transform = transforms.Lambda(lambda x: x)
        else:
            self.base_resize_transform = transforms.Compose(
                [
                    transforms.Resize(base_size),
                ]
            )

        if mean is None or std is None:
            raise ValueError("Mean and std must be specified.")
        
        self.finishing_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean, std
                ),
            ]
        )
        self.base_finishing_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean, std
                )
            ]
        )

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, base_image) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)
        # Apply the initial augmentations to the image
        if self.transform is not None:
            img = self.transform(img)
        
        base_img = self.base_finishing_transform(self.base_resize_transform(img))
        img = self.finishing_transform(self.resize_transform(img))

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, base_img

class PairedCustomCIFAR10(tv_datasets.CIFAR10):
    def __init__(self, size=-1, base_size=-1, mean=None, std=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if size == -1:
            self.resize_transform = transforms.Lambda(lambda x: x)
        else:
            self.resize_transform = transforms.Compose(
                [
                    transforms.Resize(size),
                ]
            )

        if base_size == -1:
            self.base_resize_transform = transforms.Lambda(lambda x: x)
        else:
            self.base_resize_transform = transforms.Compose(
                [
                    transforms.Resize(base_size),
                ]
            )

        if mean is None or std is None:
            raise ValueError("Mean and std must be specified.")
        
        self.finishing_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean, std
                ),
            ]
        )
        self.base_finishing_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean, std
                )
            ]
        )

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, base_image) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)
        # Apply the initial augmentations to the image
        if self.transform is not None:
            img = self.transform(img)
        
        if self.base_resize_transform is not None:
            base_img = self.base_finishing_transform(self.base_resize_transform(img))
            img = self.finishing_transform(self.resize_transform(img))
        else:
            img = self.finishing_transform(self.resize_transform(img))
            base_img = img

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, base_img

class SortedImageFolder(tv_datasets.ImageFolder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.samples.sort(key=lambda x: x[0])
        self.imgs = self.samples

class PairedImageFolder(SortedImageFolder):
    def __init__(self, size=-1, base_size=-1, mean=None, std=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if size == -1:
            self.resize_transform = transforms.Lambda(lambda x: x)
        else:
            self.resize_transform = transforms.Compose(
                [
                    transforms.Resize(size),
                ]
            )

        if base_size == -1:
            self.base_resize_transform = transforms.Lambda(lambda x: x)
        else:
            self.base_resize_transform = transforms.Compose(
                [
                    transforms.Resize(base_size),
                ]
            )

        if mean is None or std is None:
            raise ValueError("Mean and std must be specified.")

        self.base_finishing_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean, std
                ),
            ]
        )
        self.finishing_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean, std
                ),
            ]
        )

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, base_sample) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        img = self.loader(path)

        if self.transform is not None:  # initial augmentation, no resizing
            img = self.transform(img)

        base_img = self.base_finishing_transform(self.base_resize_transform(img))
        img = self.finishing_transform(self.resize_transform(img))

        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return img, target, base_img

def get_cifar(locator="cifar10/0_16", image_size=-1, base_image_size=-1, data_root="./data"):
    if locator.split("/")[0] == "cifar10":
        dataset_name = "cifar10"
        dataset_fn = PairedCustomCIFAR10
        mean, std = DATASET_FLAGS.CIFAR10_MEAN, DATASET_FLAGS.CIFAR10_STD
        if image_size == -1:
            image_size = 32
    elif locator.split("/")[0] == "cifar100" or locator.split("/")[0] == "cifar20":
        dataset_name = "cifar100"
        dataset_fn = PairedCustomCIFAR100
        mean, std = DATASET_FLAGS.CIFAR100_MEAN, DATASET_FLAGS.CIFAR100_STD
        if image_size == -1:
            image_size = 32
    else:
        raise NotImplementedError(
            f"Dataset {locator} not supported. Please use cifar10, cifar20 (cifar100 superclasses), cifar100, cinic10, or imagenet."
        )
    
    pkeep = 0.5
    experiment_idx, num_experiment = (int(n) for n in locator.split("/")[1].split("_"))

    # Create the train/test transforms, resizing is done in the dataset class
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(degrees=10),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            # transforms.RandomGrayscale(p=0.1),
            # transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        ]
    )
    transform_test = None
    transform_vanilla = None
    transform_dict = {
        "train": transform_train,
        "test": transform_test,
        "vanilla": transform_vanilla,
    }

    target_transform = None
    if locator.split("/")[0] == "cifar20":
        target_transform = lambda x: x // 5

    # Create the datasets
    private_public_dataset = dataset_fn(
        size=image_size,
        base_size=base_image_size,
        mean=mean,
        std=std,
        root=data_root,
        train=True,
        download=True,
        transform=transform_train,
        target_transform=target_transform,
    )
    test_dataset = dataset_fn(
        size=image_size,
        base_size=base_image_size,
        mean=mean,
        std=std,
        root=data_root,
        train=False,
        download=True,
        transform=transform_test,
        target_transform=target_transform,
    )

    # Save the data split
    master_keep_path = os.path.join(
        data_root, dataset_name, "{:d}".format(num_experiment), "master_keep.npy"
    )
    if os.path.exists(master_keep_path):
        master_keep = np.load(master_keep_path)
    else:
        os.makedirs(os.path.dirname(master_keep_path), exist_ok=True)
        with temp_seed(DATASET_FLAGS.DATA_SEED):
            master_keep = np.random.uniform(
                size=(num_experiment, len(private_public_dataset))
            )
            order = master_keep.argsort(0)
            master_keep = order < int(pkeep * num_experiment)
            np.save(master_keep_path, master_keep)

    if int(experiment_idx) == int(num_experiment):
        print("SPECIAL-CASING THIS IS THE FULL EVALUATION/TRAINING DATASET")
        private_indices = list(np.arange(start=0, stop=32))
        public_indices = list(np.arange(start=0, stop=len(private_public_dataset)))
    else:
        keep = np.array(master_keep[experiment_idx], dtype=bool)
        private_indices = list(np.where(keep)[0])
        public_indices = list(np.where(~keep)[0])
    
    public_dataset = Subset(private_public_dataset, public_indices)
    private_dataset = Subset(private_public_dataset, private_indices)

    return private_dataset, public_dataset, test_dataset, transform_dict

def get_cinic10(locator="cinic10/0_16", image_size=-1, base_image_size=-1, data_root="./data"):
    dataset_name = locator.split("/")[0]
    pkeep = 0.5
    experiment_idx, num_experiment = (int(n) for n in locator.split("/")[1].split("_"))

    mean, std = DATASET_FLAGS.CINIC10_MEAN, DATASET_FLAGS.CINIC10_STD # DATASET_FLAGS.CIFAR100_MEAN, DATASET_FLAGS.CIFAR100_STD #

    # Create the train/test transforms, resizing is done in the dataset class
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandAugment(num_ops=2, magnitude=9),
    ])
    transform_test = None
    transform_vanilla = None

    root_dir = os.path.join(data_root, dataset_name)

    private_public_dataset = PairedImageFolder(
        size=image_size,
        base_size=base_image_size,
        mean=mean,
        std=std,
        root=os.path.join(root_dir, "trainval"),
        transform=transform_train,
    )

    test_dataset = PairedImageFolder(
        size=image_size,
        base_size=base_image_size,
        mean=mean,
        std=std,
        root=os.path.join(root_dir, "test"),
        transform=transform_test,
    )

    master_keep_path = os.path.join(
        data_root, dataset_name, "{:d}".format(num_experiment), "master_keep.npy"
    )
    if os.path.exists(master_keep_path):
        master_keep = np.load(master_keep_path)
    else:
        os.makedirs(os.path.dirname(master_keep_path), exist_ok=True)
        with temp_seed(DATASET_FLAGS.DATA_SEED):
            master_keep = np.random.uniform(
                size=(num_experiment, len(private_public_dataset))
            )
        order = master_keep.argsort(0)
        master_keep = order < int(pkeep * num_experiment)
        np.save(master_keep_path, master_keep)

    if int(experiment_idx) == int(num_experiment):
        print("SPECIAL-CASING THIS IS THE FULL EVALUATION/TRAINING DATASET")
        private_indices = list(np.arange(start=0, stop=32))
        public_indices = list(np.arange(start=0, stop=len(private_public_dataset)))
    else:
        keep = np.array(master_keep[experiment_idx], dtype=bool)
        private_indices = list(np.where(keep)[0])
        public_indices = list(np.where(~keep)[0])
    
    public_dataset = Subset(private_public_dataset, public_indices)
    private_dataset = Subset(private_public_dataset, private_indices)

    transform_dict = {
        "train": transform_train,
        "test": transform_test,
        "vanilla": transform_vanilla,
    }

    return private_dataset, public_dataset, test_dataset, transform_dict

def get_imagenet(
    locator="imagenet-1k/0_16", image_size=-1, base_image_size=-1, data_root="./data",
):
    mean = DATASET_FLAGS.IMAGENET_MEAN
    std = DATASET_FLAGS.IMAGENET_STD

    experiment_idx, num_experiment = (int(n) for n in locator.split("/")[1].split("_"))
    dataset_name = locator.split("/")[0]

    # Get train/test transforms
    transform_train = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomCrop(256, padding=32),
            transforms.RandomHorizontalFlip(),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
        ]
    )
    transform_vanilla = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
        ]
    )

    root_dir = os.path.join(data_root, dataset_name)
    
    full_dataset = PairedImageFolder(
        size=image_size,
        base_size=base_image_size,
        mean=mean,
        std=std,
        root=os.path.join(root_dir, "train"),
        transform=transform_train,
    )
    full_dataset_test = PairedImageFolder(
        size=image_size,
        base_size=base_image_size,
        mean=mean,
        std=std,
        root=os.path.join(root_dir, "train"),
        transform=transform_test,
    )

    # get private/public/test split for experiment
    pkeep_private = 0.4
    pkeep_public = 0.4

    master_keep_path = os.path.join(
        data_root, dataset_name, "{:d}".format(num_experiment), "master_keep.npy"
    )
    if os.path.exists(master_keep_path):
        master_keep = np.load(master_keep_path)
    else:
        os.makedirs(os.path.dirname(master_keep_path), exist_ok=True)
        with temp_seed(DATASET_FLAGS.DATA_SEED):
            master_keep = np.random.uniform(
                size=(num_experiment, len(full_dataset)), low=0, high=1
            )
        order = master_keep.argsort(0)

        master_keep = np.zeros((num_experiment, len(full_dataset)), dtype=int)
        split1_threshold = int(pkeep_private * num_experiment)
        master_keep[order < split1_threshold] = 0

        split2_threshold = int((pkeep_private + pkeep_public) * num_experiment)
        master_keep[(order >= split1_threshold) & (order < split2_threshold)] = 1

        master_keep[order >= split2_threshold] = 2
        np.save(master_keep_path, master_keep)

    keep = np.array(master_keep[experiment_idx], dtype=int)
    private_indices = list(np.where(keep == 0)[0])  # First 40%
    public_indices = list(np.where(keep == 1)[0])  # Next 40%
    test_indices = list(np.where(keep == 2)[0])  # Final 20%

    public_dataset = Subset(full_dataset, public_indices)
    private_dataset = Subset(full_dataset, private_indices)
    test_dataset = Subset(full_dataset_test, test_indices)

    transform_dict = {
        "train": transform_train,
        "test": transform_test,
        "vanilla": transform_vanilla,
    }

    return private_dataset, public_dataset, test_dataset, transform_dict

def get_data(
    split_frac: float,
    dataset: str,
    image_size: int,
    base_image_size: int,
    data_root: str,
    cls_drop: Optional[list] = None,
    cls_samples: Optional[int] = None,
):
    """
    Get the dataset and transforms for the given dataset name.
    """
    if dataset.startswith("cifar"):
        (
            private_dataset,
            public_dataset,
            test_dataset,
            transform_dict,
        ) = get_cifar(locator=dataset, image_size=image_size, base_image_size=base_image_size, data_root=data_root)
    elif dataset.startswith("cinic10"):
        (
            private_dataset,
            public_dataset,
            test_dataset,
            transform_dict,
        ) = get_cinic10(locator=dataset, image_size=image_size, base_image_size=base_image_size, data_root=data_root)
    elif dataset.startswith("imagenet"):
        (
            private_dataset,
            public_dataset,
            test_dataset,
            transform_dict,
        ) = get_imagenet(locator=dataset, image_size=image_size, base_image_size=base_image_size, data_root=data_root)
    else:
        raise NotImplementedError(
            f"Dataset {dataset} not supported. Please use cifar10, cifar100, cinic10, or imagenet."
        )
    full_dataset = private_dataset
    # Split the training dataset into private (%=split_frac) and public (%=1-split_frac) datasets.
    if public_dataset is None:
        with temp_seed(DATASET_FLAGS.DATA_SEED):
            indices = np.random.permutation(len(full_dataset))

            th_indices = int(len(indices * split_frac))
            indices_a = indices[:th_indices]
            indices_b = indices[th_indices:]

            public_dataset = Subset(full_dataset, indices_b)
            private_dataset = Subset(full_dataset, indices_a)

    # Drop classes if specified
    current_indices = public_dataset.indices
    original_dataset = public_dataset.dataset

    if cls_drop is not None and len(cls_drop) > 0:
        cls_drop = set(cls_drop)
        updated_indices = [
            i
            for i in current_indices
            if original_dataset.targets[i] not in cls_drop
        ]
        current_indices = updated_indices

    if cls_samples is not None:
        updated_indices = []
        classes = set(original_dataset.targets) if cls_drop is None else set(original_dataset.targets) - set(cls_drop)
        # For each class in the dataset, select only cls_samples indices from current_indices
        for cls in set(classes):
            # Get indices in public_dataset that belong to this class
            cls_indices = [i for i in current_indices if original_dataset.targets[i] == cls]
            if len(cls_indices) > cls_samples:
                selected = np.random.choice(cls_indices, size=cls_samples, replace=False)
                updated_indices.extend(selected.tolist())
            else:
                updated_indices.extend(cls_indices)
        # Optionally, sort indices if order matters
        updated_indices.sort()
        current_indices = updated_indices

    public_dataset = Subset(
        original_dataset, current_indices
    )

    dataset_dict = {
        "private": private_dataset,
        "public": public_dataset,
        "test": test_dataset,
    }
    return dataset_dict, transform_dict

class CustomDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name,
        stage,
        batch_size: int = 16,
        num_workers: int = 16,
        image_size: int = -1,
        base_image_size: int = -1,
        data_root: str = "./data",
        use_augmentation: bool = True,
        cls_drop: Optional[list] = None,
        cls_samples: Optional[int] = None,
    ):
        super().__init__()

        self.dataset_name = dataset_name
        self.stage = stage
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.base_image_size = base_image_size
        self.data_root = data_root
        self.use_augmentation = use_augmentation
        self.cls_drop = cls_drop
        self.cls_samples = cls_samples

    def prepare_data(self) -> None:
        # TODO: Download the dataset if needed
        pass

    def setup(self, stage: Optional[str] = None) -> None:

        dataset_dict, transform_dict = get_data(
            split_frac=0.5,
            dataset=self.dataset_name,
            image_size=self.image_size,
            base_image_size=self.base_image_size,
            data_root=self.data_root,
            cls_drop=self.cls_drop,
            cls_samples=self.cls_samples,
        )

        stage = self.stage if self.stage is not None else stage
        if stage == "base":
            print("Base stage data")
            # Base model is trained on the private dataset
            self.train_dataset = dataset_dict["private"]
            self.val_dataset = dataset_dict["test"]
            self.test_dataset = dataset_dict["public"]
        elif stage == "mia":
            print("MIA stage data")
            # MIA model is trained on the public dataset
            self.train_dataset = dataset_dict["public"]
            self.val_dataset = dataset_dict["test"]
            self.test_dataset = dataset_dict["private"]
        elif stage == "eval":
            print("Eval stage data")
            # For evaluating the MIA model, test_dataset contains the base model train data and val_dataset contains heldout public data.
            self.train_dataset = dataset_dict["public"]
            self.val_dataset = dataset_dict["test"]
            self.test_dataset = dataset_dict["private"]
            # Remove any training transforms
            self.train_dataset = set_transform(
                self.train_dataset, transform_dict["vanilla"]
            )
            self.test_dataset = set_transform(
                self.test_dataset, transform_dict["vanilla"]
            )
            self.val_dataset = set_transform(
                self.val_dataset, transform_dict["vanilla"]
            )
        else:
            raise ValueError(f"Mode {stage} not recognized. Use 'base', 'mia', or 'eval'.")
        
        return
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
        )
    
    def predict_dataloader(self) -> DataLoader:
        return [
            DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=self.num_workers,
            ),
            DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=self.num_workers,
            ),
        ]
