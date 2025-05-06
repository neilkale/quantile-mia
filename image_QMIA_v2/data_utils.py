import contextlib
import os
from typing import NamedTuple, Optional, Tuple, Any
from PIL import Image, ImageFile

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import torchvision.datasets as tv_datasets

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
    def __init__(self, size=-1, mean=None, std=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if size == -1:
            size = 32

        self.resize_transform = transforms.Compose(
            [
                transforms.Resize(size),
            ]
        )
        self.finishing_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean if mean is not None else DATASET_FLAGS.CIFAR10_MEAN,
                    std if std is not None else DATASET_FLAGS.CIFAR10_STD,
                ),
            ]
        )

        if size != 32:
            self.base_resize_transform = transforms.Compose(
                [
                    transforms.Resize(32),
                ]
            )
            self.base_finishing_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        DATASET_FLAGS.CIFAR10_MEAN,
                        DATASET_FLAGS.CIFAR10_STD,
                    )
                ]
            )
        else:
            self.base_resize_transform = None
            self.base_finishing_transform = None

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

class PairedCustomCIFAR10(tv_datasets.CIFAR10):
    def __init__(self, size=-1, mean=None, std=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if size == -1:
            size = 32

        self.resize_transform = transforms.Compose(
            [
                transforms.Resize(size),
            ]
        )
        self.finishing_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean if mean is not None else DATASET_FLAGS.CIFAR10_MEAN,
                    std if std is not None else DATASET_FLAGS.CIFAR10_STD,
                ),
            ]
        )

        if size != 32:
            self.base_resize_transform = transforms.Compose(
                [
                    transforms.Resize(32),
                ]
            )
            self.base_finishing_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        DATASET_FLAGS.CIFAR10_MEAN,
                        DATASET_FLAGS.CIFAR10_STD,
                    )
                ]
            )
        else:
            self.base_resize_transform = None
            self.base_finishing_transform = None

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

class PairedImageFolder(tv_datasets.ImageFolder):
    def __init__(self, size=-1, mean=None, std=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if size == -1:
            size = 32

        self.resize_transform = transforms.Compose(
            [
                transforms.Resize(size),
            ]
        )
        if size != 32:
            self.base_resize_transform = transforms.Compose(
                [
                    transforms.Resize(32),
                ]
            )
        else:
            self.base_resize_transform = None

        self.base_finishing_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    DATASET_FLAGS.CIFAR10_MEAN, DATASET_FLAGS.CIFAR10_STD
                ),
            ]
        )
        self.finishing_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean if mean is not None else DATASET_FLAGS.CIFAR10_MEAN,
                    std if std is not None else DATASET_FLAGS.CIFAR10_STD,
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
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.base_resize_transform is not None:
            base_img = self.base_finishing_transform(self.base_resize_transform(img))
            img = self.finishing_transform(self.resize_transform(img))
        else:
            img = self.finishing_transform(self.resize_transform(img))
            base_img = img

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, base_img

def get_cifar(locator="cifar10/0_16", image_size=-1, data_root="./data"):
    if locator.split("/")[0] == "cifar10":
        dataset_name = "cifar10"
        dataset_fn = PairedCustomCIFAR10
        mean, std = DATASET_FLAGS.CIFAR10_MEAN, DATASET_FLAGS.CIFAR10_STD
        if image_size == -1:
            image_size = 32
    elif locator.split("/")[0] == "cifar100":
        dataset_name = "cifar100"
        dataset_fn = PairedCustomCIFAR100
        mean, std = DATASET_FLAGS.CIFAR100_MEAN, DATASET_FLAGS.CIFAR100_STD
        if image_size == -1:
            image_size = 32
    else:
        raise NotImplementedError(
            f"Dataset {locator} not supported. Please use cifar10, cifar100, cinic10, or imagenet."
        )
    
    pkeep = 0.5
    experiment_idx, num_experiment = (int(n) for n in locator.split("/")[1].split("_"))

    # Create the train/test transforms, resizing is done in the dataset class
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()]
    )
    transform_test = None
    transform_vanilla = None
    transform_dict = {
        "train": transform_train,
        "test": transform_test,
        "vanilla": transform_vanilla,
    }

    # Create the datasets
    private_public_dataset = dataset_fn(
        size=image_size,
        mean=mean,
        std=std,
        root=data_root,
        train=True,
        download=True,
        transform=transform_train,
    )
    test_dataset = dataset_fn(
        size=image_size,
        mean=mean,
        std=std,
        root=data_root,
        train=False,
        download=True,
        transform=transform_test,
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

def get_cinic10(locator="cinic10/0_16", image_size=-1, data_root="./data"):
    dataset_name = locator.split("/")[0]
    pkeep = 0.5
    experiment_idx, num_experiment = (int(n) for n in locator.split("/")[1].split("_"))

    if image_size == -1:
        image_size = 32
    mean, std = DATASET_FLAGS.CINIC10_MEAN, DATASET_FLAGS.CINIC10_STD

    # Create the train/test transforms, resizing is done in the dataset class
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()]
    )
    transform_test = None
    transform_vanilla = None

    root_dir = os.path.join(data_root, dataset_name)

    private_public_dataset = PairedImageFolder(
        size=image_size,
        mean=mean,
        std=std,
        root=os.path.join(root_dir, "trainval"),
        transform=transform_train,
    )

    test_dataset = PairedImageFolder(
        size=image_size,
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
    locator="imagenet-1k/0_16", image_size=-1, data_root="./data",
):
    mean = DATASET_FLAGS.IMAGENET_MEAN
    std = DATASET_FLAGS.IMAGENET_STD

    experiment_idx, num_experiment = (int(n) for n in locator.split("/")[1].split("_"))
    dataset_name = locator.split("/")[0]

    if image_size == -1:
        image_size = 224
    resize_size = int(256.0 / 224 * image_size)

    # Get train/test transforms
    transform_train = transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    transform_dict = {
        "train": transform_train,
        "test": transform_test,
        "vanilla": transform_test,
    }

    root_dir = os.path.join(data_root, dataset_name)
    private_public_dataset = tv_datasets.ImageFolder(
        os.path.join(root_dir, "train"), transform=transform_train
    )
    test_dataset = tv_datasets.ImageFolder(
        os.path.join(root_dir, "val"), transform=transform_test
    )

    # get private/public split for experiment
    pkeep = 0.5

    master_keep_path = os.path.join(
        data_root, dataset_name, "{:d}".format(num_experiment), "master_keep.npy"
    )
    if os.path.exists(master_keep_path):
        master_keep = np.load(master_keep_path)
    else:
        os.makedirs(os.path.dirname(master_keep_path), exist_ok=True)
        with temp_seed(DATASET_FLAGS.DATA_SEED):
            master_keep = np.random.uniform(
                size=(num_experiment, len(private_public_dataset)), low=0, high=1
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

def get_data(
    split_frac: float,
    dataset: str,
    image_size: int,
    data_root: str,
    cls_drop: Optional[list] = None,
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
        ) = get_cifar(locator=dataset, image_size=image_size, data_root=data_root)
    elif dataset.startswith("cinic10"):
        (
            private_dataset,
            public_dataset,
            test_dataset,
            transform_dict,
        ) = get_cinic10(locator=dataset, image_size=image_size, data_root=data_root)
    elif dataset.startswith("imagenet"):
        (
            private_dataset,
            public_dataset,
            test_dataset,
            transform_dict,
        ) = get_imagenet(locator=dataset, image_size=image_size, data_root=data_root)
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
    if cls_drop is not None and len(cls_drop) > 0:
        cls_drop = set(cls_drop)

        original_dataset = public_dataset.dataset
        current_indices = public_dataset.indices
        new_indices = [
            i
            for i in current_indices
            if original_dataset.targets[i] not in cls_drop
        ]
        public_dataset = Subset(
            original_dataset, new_indices
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
        data_root: str = "./data",
        use_augmentation: bool = True,
        cls_drop: Optional[list] = None,
    ):
        super().__init__()

        self.dataset_name = dataset_name
        self.stage = stage
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.data_root = data_root
        self.use_augmentation = use_augmentation
        self.cls_drop = cls_drop

    def prepare_data(self) -> None:
        # TODO: Download the dataset if needed
        pass

    def setup(self, stage: Optional[str] = None) -> None:

        dataset_dict, transform_dict = get_data(
            split_frac=0.5,
            dataset=self.dataset_name,
            image_size=self.image_size,
            data_root=self.data_root,
            cls_drop=self.cls_drop,
        )

        stage = self.stage if self.stage is not None else stage
        if stage == "base":
            # Base model is trained on the private dataset
            self.train_dataset = dataset_dict["private"]
            self.val_dataset = dataset_dict["test"]
            self.test_dataset = dataset_dict["public"]
        elif stage == "mia":
            # MIA model is trained on the public dataset
            self.train_dataset = dataset_dict["public"]
            self.val_dataset = dataset_dict["test"]
            self.test_dataset = dataset_dict["private"]
        elif stage == "eval":
            # For evaluating the MIA model, test_dataset contains the base model train data and val_dataset contains heldout public data.
            self.train_dataset = dataset_dict["public"]
            self.test_dataset = dataset_dict["private"]
            self.val_dataset = dataset_dict["test"]
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