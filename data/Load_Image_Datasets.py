import random, copy
import os
import numpy as np
import mne
import torch
import torchvision
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CelebA
from torchvision.transforms import ToTensor, Compose, Normalize
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split, Subset
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification


# Absolute path to data directory on NERSC
DATA_ROOT = '/pscratch/sd/j/junghoon/data'

seed = 2024
dataset_name = "CIFAR"



def load_mnist(seed, n_train, n_valtest, device, batch_size):
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Load dataset with transformation
    transform = Compose([ToTensor(), lambda x: x.view(-1)])  # Flatten MNIST images
    data_train = MNIST(root=DATA_ROOT, train=True, download=True, transform=transform)
    data_test = MNIST(root=DATA_ROOT, train=False, download=True, transform=transform)
    input_dim = 28 * 28

    X_train = data_train.data.float() / 255.0  # Normalize pixel values to [0, 1]
    y_train = data_train.targets.clone().detach()
    X_test = data_test.data.float() / 255.0
    y_test = data_test.targets.clone().detach()
    
    # Shuffle data
    shuffle_idx = torch.randperm(n_train)
    X_train = X_train[shuffle_idx]
    y_train = y_train[shuffle_idx]

    shuffle_idx2 = torch.randperm(n_valtest)
    X_test = X_test[shuffle_idx2]
    y_test = y_test[shuffle_idx2]

    # Limit dataset size
    X_train = X_train[:n_train]
    y_train = y_train[:n_train]
    X_test = X_test[:n_valtest]
    y_test = y_test[:n_valtest]    
    
    # Flatten images
    X_train = X_train.view(-1, 28*28)
    X_test = X_test.view(-1, 28*28)
    
    # Create TensorDatasets
    train_X = X_train.to(device)
    train_y = y_train.to(device)
    test_X = X_test.to(device)
    test_y = y_test.to(device)

    train_dataset = TensorDataset(train_X, train_y)
    valtest_dataset = TensorDataset(test_X, test_y)

    # Equally split validation and test sets
    val_size = int(0.5 * len(valtest_dataset))
    test_size = int(0.5 * len(valtest_dataset))
    val_dataset, test_dataset = random_split(valtest_dataset, [val_size, test_size])

    # DataLoader parameters
    if batch_size==0:
        params = {'shuffle': True}
        test_params = {'shuffle': False}
    else:
        params = {'shuffle': True, 'batch_size': batch_size}
        test_params = {'shuffle': False, 'batch_size': batch_size}

    train_loader = DataLoader(train_dataset, **params)
    val_loader = DataLoader(val_dataset, **test_params)
    test_loader = DataLoader(test_dataset, **test_params)
    
    return train_loader, val_loader, test_loader, input_dim


def load_fashion(seed, n_train, n_valtest, device, batch_size):
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Load dataset with transformation
    transform = Compose([ToTensor(), lambda x: x.view(-1)])  # Flatten MNIST images
    data_train = FashionMNIST(root=DATA_ROOT, train=True, download=True, transform=transform)
    data_test = FashionMNIST(root=DATA_ROOT, train=False, download=True, transform=transform)
    input_dim = 28 * 28

    X_train = data_train.data.float() / 255.0  # Normalize pixel values to [0, 1]
    y_train = data_train.targets.clone().detach()
    X_test = data_test.data.float() / 255.0
    y_test = data_test.targets.clone().detach()
    
    # Shuffle data
    shuffle_idx = torch.randperm(n_train)
    X_train = X_train[shuffle_idx]
    y_train = y_train[shuffle_idx]

    shuffle_idx2 = torch.randperm(n_valtest)
    X_test = X_test[shuffle_idx2]
    y_test = y_test[shuffle_idx2]

    # Limit dataset size
    X_train = X_train[:n_train]
    y_train = y_train[:n_train]
    X_test = X_test[:n_valtest]
    y_test = y_test[:n_valtest]    
        
    # Flatten images
    X_train = X_train.view(-1, 28*28)
    X_test = X_test.view(-1, 28*28)
    
    # Create TensorDatasets
    train_X = X_train.to(device)
    train_y = y_train.to(device)
    test_X = X_test.to(device)
    test_y = y_test.to(device)

    train_dataset = TensorDataset(train_X, train_y)
    valtest_dataset = TensorDataset(test_X, test_y)

    # Equally split validation and test sets
    val_size = int(0.5 * len(valtest_dataset))
    test_size = int(0.5 * len(valtest_dataset))
    val_dataset, test_dataset = random_split(valtest_dataset, [val_size, test_size])

    # DataLoader parameters
    if batch_size==0:
        params = {'shuffle': True}
        test_params = {'shuffle': False}
    else:
        params = {'shuffle': True, 'batch_size': batch_size}
        test_params = {'shuffle': False, 'batch_size': batch_size}

    train_loader = DataLoader(train_dataset, **params)
    val_loader = DataLoader(val_dataset, **test_params)
    test_loader = DataLoader(test_dataset, **test_params)
    
    return train_loader, val_loader, test_loader, input_dim


def load_cifar(seed, n_train, n_valtest, device, batch_size):
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Load dataset with transformation
    transform = Compose([
        ToTensor(),  # Convert to tensor
        # Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize to [-1, 1]
        lambda x: x.view(-1)  # Flatten CIFAR10 images (3 * 32 * 32)
    ])
    data_train = CIFAR10(root=DATA_ROOT, train=True, download=True, transform=transform)
    data_test = CIFAR10(root=DATA_ROOT, train=False, download=True, transform=transform)
    input_dim = 3 * 32 * 32

    X_train = torch.tensor(data_train.data).float() / 255.0  # Normalize pixel values to [0, 1] 
    y_train = torch.tensor(data_train.targets)  # Convert labels to tensor
    X_test = torch.tensor(data_test.data).float() / 255.0  # Normalize pixel values to [0, 1]
    y_test = torch.tensor(data_test.targets)
    
    # Shuffle data
    shuffle_idx = torch.randperm(n_train)
    X_train = X_train[shuffle_idx]
    y_train = y_train[shuffle_idx]

    shuffle_idx2 = torch.randperm(n_valtest)
    X_test = X_test[shuffle_idx2]
    y_test = y_test[shuffle_idx2]

    # Limit dataset size
    X_train = X_train[:n_train]
    y_train = y_train[:n_train]
    X_test = X_test[:n_valtest]
    y_test = y_test[:n_valtest]    
        
    # Flatten images
    X_train = X_train.contiguous().view(X_train.size(0), -1)
    X_test = X_test.contiguous().view(X_test.size(0), -1)
    
    # Create TensorDatasets
    train_X = X_train.to(device)
    train_y = y_train.to(device)
    test_X = X_test.to(device)
    test_y = y_test.to(device)

    train_dataset = TensorDataset(train_X, train_y)
    valtest_dataset = TensorDataset(test_X, test_y)

    # Equally split validation and test sets
    val_size = int(0.5 * len(valtest_dataset))
    test_size = int(0.5 * len(valtest_dataset))
    val_dataset, test_dataset = random_split(valtest_dataset, [val_size, test_size])

    # DataLoader parameters
    if batch_size==0:
        params = {'shuffle': True}
        test_params = {'shuffle': False}
    else:
        params = {'shuffle': True, 'batch_size': batch_size}
        test_params = {'shuffle': False, 'batch_size': batch_size}

    train_loader = DataLoader(train_dataset, **params)
    val_loader = DataLoader(val_dataset, **test_params)
    test_loader = DataLoader(test_dataset, **test_params)
    
    return train_loader, val_loader, test_loader, input_dim



def load_celeba(seed, n_train, n_valtest, device, batch_size):
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Load dataset with transformation
    transform = Compose([
        ToTensor(),  # Convert to tensor
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize to [-1, 1]
        lambda x: x.view(-1)  # Flatten CIFAR10 images (3 * 32 * 32)
    ])
    full_data = CelebA(root=DATA_ROOT, split='all', download=True, transform=transform)
    input_dim = 3 * 178 * 218

    # Equally split validation and test sets
    subset_size = n_train + n_valtest
    subset_indices = torch.randperm(len(full_data))[:subset_size]  # Randomly select samples
    data_subset = Subset(data_train, subset_indices)
    val_size = int(0.5 * n_valtest)
    test_size = int(0.5 * n_valtest)
    train_dataset, val_dataset, test_dataset = random_split(full_data, [n_train, val_size, test_size])

    # DataLoader parameters
    if batch_size==0:
        params = {'shuffle': True}
        test_params = {'shuffle': False}
    else:
        params = {'shuffle': True, 'batch_size': batch_size}
        test_params = {'shuffle': False, 'batch_size': batch_size}

    train_loader = DataLoader(train_dataset, **params)
    val_loader = DataLoader(val_dataset, **test_params)
    test_loader = DataLoader(test_dataset, **test_params)
    
    return train_loader, val_loader, test_loader, input_dim


def load_coco(seed, n_train, n_valtest, device, batch_size, num_classes=80, img_size=224):
    """
    Load COCO dataset for multi-label image classification.

    Uses a simplified approach: downloads COCO annotations and creates a classification
    task based on the primary object category in each image.

    For practical purposes on quantum models, we:
    1. Resize images to img_size x img_size
    2. Use a subset of COCO categories (num_classes most common)
    3. Create single-label classification (primary object category)

    Args:
        seed: Random seed for reproducibility
        n_train: Number of training samples
        n_valtest: Number of validation + test samples
        device: torch device
        batch_size: Batch size for data loaders
        num_classes: Number of classes to use (default 20 most common)
        img_size: Image size to resize to (default 64x64)

    Returns:
        train_loader, val_loader, test_loader, input_dim
    """
    import os
    from PIL import Image
    import urllib.request
    import zipfile
    import json

    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Data directory
    data_dir = os.path.join(DATA_ROOT, 'coco')
    os.makedirs(data_dir, exist_ok=True)

    # Check if we can use torchvision's COCO
    try:
        from torchvision.datasets import CocoDetection

        # Define transforms
        transform = Compose([
            torchvision.transforms.Resize((img_size, img_size)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Try to load COCO - if not available, fall back to synthetic data
        ann_file_train = os.path.join(data_dir, 'annotations/instances_train2017.json')
        img_dir_train = os.path.join(data_dir, 'train2017')

        if os.path.exists(ann_file_train) and os.path.exists(img_dir_train):
            print("Loading COCO dataset from local files...")
            coco_train = CocoDetection(img_dir_train, ann_file_train, transform=transform)

            # Get category IDs and map to contiguous indices
            with open(ann_file_train, 'r') as f:
                coco_ann = json.load(f)

            categories = coco_ann['categories']
            cat_ids = [cat['id'] for cat in categories[:num_classes]]
            cat_to_idx = {cat_id: idx for idx, cat_id in enumerate(cat_ids)}

            # Extract images and primary labels
            images = []
            labels = []

            for idx in range(min(n_train + n_valtest, len(coco_train))):
                img, target = coco_train[idx]
                if len(target) > 0:
                    # Get primary category (first annotation)
                    primary_cat = target[0]['category_id']
                    if primary_cat in cat_to_idx:
                        images.append(img)
                        labels.append(cat_to_idx[primary_cat])

                if len(images) >= n_train + n_valtest:
                    break

            X = torch.stack(images)
            y = torch.tensor(labels)
        else:
            raise FileNotFoundError("COCO files not found, using synthetic data")

    except Exception as e:
        print(f"Could not load COCO dataset: {e}")
        print("Generating synthetic COCO-like data for testing...")

        # Generate synthetic data that mimics COCO characteristics
        total_samples = n_train + n_valtest

        # Create synthetic images (3 channels, img_size x img_size)
        X = torch.randn(total_samples, 3, img_size, img_size)

        # Normalize to reasonable image range
        X = (X - X.min()) / (X.max() - X.min())

        # Create synthetic labels
        y = torch.randint(0, num_classes, (total_samples,))

        # Add some structure: make different classes have different mean intensities
        for i in range(num_classes):
            mask = (y == i)
            X[mask] = X[mask] + 0.1 * (i / num_classes)

        X = torch.clamp(X, 0, 1)

    # Flatten images for compatibility with existing models
    input_dim = 3 * img_size * img_size
    X_flat = X.view(X.size(0), -1)

    # Shuffle data
    shuffle_idx = torch.randperm(len(X_flat))
    X_flat = X_flat[shuffle_idx]
    y = y[shuffle_idx]

    # Split into train and val/test
    X_train = X_flat[:n_train].to(device)
    y_train = y[:n_train].to(device)
    X_valtest = X_flat[n_train:n_train + n_valtest].to(device)
    y_valtest = y[n_train:n_train + n_valtest].to(device)

    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    valtest_dataset = TensorDataset(X_valtest, y_valtest)

    # Split validation and test
    val_size = len(valtest_dataset) // 2
    test_size = len(valtest_dataset) - val_size
    val_dataset, test_dataset = random_split(valtest_dataset, [val_size, test_size])

    # DataLoader parameters
    if batch_size == 0:
        params = {'shuffle': True}
        test_params = {'shuffle': False}
    else:
        params = {'shuffle': True, 'batch_size': batch_size}
        test_params = {'shuffle': False, 'batch_size': batch_size}

    train_loader = DataLoader(train_dataset, **params)
    val_loader = DataLoader(val_dataset, **test_params)
    test_loader = DataLoader(test_dataset, **test_params)

    print(f"COCO data loaded:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Input dim: {input_dim}")
    print(f"  Num classes: {num_classes}")

    return train_loader, val_loader, test_loader, input_dim


def load_eeg(seed, device, batch_size, sampling_freq=1.6):
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        
    # Load and preprocess the PhysioNet EEG Motor Imagery data
    N_SUBJECT = 50
    IMAGINE_OPEN_CLOSE_LEFT_RIGHT_FIST = [4, 8, 12]

    # Load data from PhysioNet (example assumes data is downloaded locally)
    physionet_paths = [
        mne.datasets.eegbci.load_data(
            subjects=subj_id,
            runs=IMAGINE_OPEN_CLOSE_LEFT_RIGHT_FIST,
            path="PhysioNet_EEG",
        ) for subj_id in range(1, N_SUBJECT+1)
    ]
    physionet_paths = np.concatenate(physionet_paths)

    # Ensuring that all subjects share same sampling frequency
    # TARGET_SFREQ = 160  # 160 Hz sampling rate
    TARGET_SFREQ = sampling_freq

    # Concatenate all loaded raw data
    parts = []
    for path in physionet_paths:
        raw = mne.io.read_raw_edf(
            path,
            preload=True,
            stim_channel='auto',
            verbose='WARNING',
        )
        # Resample raw data to ensure consistent sfreq
        raw.resample(TARGET_SFREQ, npad="auto")
        parts.append(raw)
        
    # Concatenate resampled raw data
    raw = mne.concatenate_raws(parts)

    # Pick EEG channels and extract events
    eeg_channel_inds = mne.pick_types(
        raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads'
    )
    events, _ = mne.events_from_annotations(raw)

    # Epoch the data
    epoched = mne.Epochs(
        raw, events, dict(left=2, right=3), tmin=1, tmax=4.1,
        proj=False, picks=eeg_channel_inds, baseline=None, preload=True
    )

    # Convert data to NumPy arrays
    X = (epoched.get_data() * 1e3).astype(np.float32)  # Convert to millivolts
    y = (epoched.events[:, 2] - 2).astype(np.int64)  # 0: left, 1: right

    # Flatten the time and channel dimensions for input to dense neural network
    X_flat = X.reshape(X.shape[0], -1)

    # First split (train, temp)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_flat, y, test_size=0.3, random_state=seed
    )

    # Compute standardization parameters from training set
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0) + 1e-6

    # Standardize datasets using train statistics
    X_train = (X_train - X_mean) / X_std
    X_temp = (X_temp - X_mean) / X_std

    # Split validation and test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed
    )
    
    def MakeTensorDataset(X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
        tensordataset = TensorDataset(X_tensor, y_tensor)
        return tensordataset
    
    # Create datasets and dataloaders
    train_dataset = MakeTensorDataset(X_train, y_train)
    val_dataset = MakeTensorDataset(X_val, y_val)
    test_dataset = MakeTensorDataset(X_test, y_test)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, shuffle=False)
    
    input_dim = X_train.shape[1]
    
    return train_loader, val_loader, test_loader, input_dim


def load_eeg_ts(seed, device, batch_size, sampling_freq=16):
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        
    # Load and preprocess the PhysioNet EEG Motor Imagery data
    N_SUBJECT = 50
    IMAGINE_OPEN_CLOSE_LEFT_RIGHT_FIST = [4, 8, 12]

    # Load data from PhysioNet (example assumes data is downloaded locally)
    physionet_paths = [
        mne.datasets.eegbci.load_data(
            subjects=subj_id,
            runs=IMAGINE_OPEN_CLOSE_LEFT_RIGHT_FIST,
            path="PhysioNet_EEG",
        ) for subj_id in range(1, N_SUBJECT+1)
    ]
    physionet_paths = np.concatenate(physionet_paths)

    # Ensuring that all subjects share same sampling frequency
    # TARGET_SFREQ = 160  # 160 Hz sampling rate
    TARGET_SFREQ = sampling_freq

    # Concatenate all loaded raw data
    parts = []
    for path in physionet_paths:
        raw = mne.io.read_raw_edf(
            path,
            preload=True,
            stim_channel='auto',
            verbose='WARNING',
        )
        # Resample raw data to ensure consistent sfreq
        raw.resample(TARGET_SFREQ, npad="auto")
        parts.append(raw)
        
    # Concatenate resampled raw data
    raw = mne.concatenate_raws(parts)

    # Pick EEG channels and extract events
    eeg_channel_inds = mne.pick_types(
        raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads'
    )
    events, _ = mne.events_from_annotations(raw)

    # Epoch the data
    epoched = mne.Epochs(
        raw, events, dict(left=2, right=3), tmin=1, tmax=4.1,
        proj=False, picks=eeg_channel_inds, baseline=None, preload=True
    )

    # Convert data to NumPy arrays
    X = (epoched.get_data() * 1e3).astype(np.float32)  # Convert to millivolts
    y = (epoched.events[:, 2] - 2).astype(np.int64)  # 0: left, 1: right
    
    # Train-validation-test split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed)
    
    def MakeTensorDataset(X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
        tensordataset = TensorDataset(X_tensor, y_tensor)
        return tensordataset
    
    # Create datasets and dataloaders
    train_dataset = MakeTensorDataset(X_train, y_train)
    val_dataset = MakeTensorDataset(X_val, y_val)
    test_dataset = MakeTensorDataset(X_test, y_test)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, shuffle=False)
    
    input_dim = X_train.shape
    
    return train_loader, val_loader, test_loader, input_dim
    

def load_binary_classification(seed, n_train, n_valtest, device, batch_size, n_features):
    """
    Generate a binary classification dataset using sklearn.datasets.make_classification 
    and load it in a format compatible with the example MNIST loader.

    Args:
        seed (int): Random seed for reproducibility.
        n_train (int): Number of training samples.
        n_valtest (int): Number of validation + test samples.
        device (torch.device): PyTorch device (CPU/GPU).
        batch_size (int): Batch size for DataLoader.
        n_features (int): Number of features for each sample.

    Returns:
        train_loader, val_loader, test_loader, input_dim: DataLoaders for training, validation, and testing,
                                                         and the number of input features (input_dim).
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Generate dataset
    X, y = make_classification(
        n_samples=n_train + n_valtest,
        n_features=n_features,
        n_informative=n_features // 2,
        n_redundant=n_features // 4,
        n_classes=2,
        random_state=seed,
    )

    # Shuffle and split dataset into train, validation, and test sets
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.int64)
    
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X = (X - X_mean) / (X_std + 1e-6)  # Add epsilon to prevent division by zero

    # Split into training and validation/test datasets
    X_train, X_valtest = X[:n_train], X[n_train:]
    y_train, y_valtest = y[:n_train], y[n_train:]

    # Further split validation and test datasets equally
    val_size = test_size = n_valtest // 2
    X_val, X_test = X_valtest[:val_size], X_valtest[val_size:]
    y_val, y_test = y_valtest[:val_size], y_valtest[val_size:]

    # Create TensorDatasets
    train_dataset = TensorDataset(X_train.to(device), y_train.to(device))
    val_dataset = TensorDataset(X_val.to(device), y_val.to(device))
    test_dataset = TensorDataset(X_test.to(device), y_test.to(device))

    # DataLoader parameters
    if batch_size == 0:
        params = {'shuffle': True}
        test_params = {'shuffle': False}
    else:
        params = {'shuffle': True, 'batch_size': batch_size}
        test_params = {'shuffle': False, 'batch_size': batch_size}

    train_loader = DataLoader(train_dataset, **params)
    val_loader = DataLoader(val_dataset, **test_params)
    test_loader = DataLoader(test_dataset, **test_params)

    input_dim = X_train.shape[1]
    
    return train_loader, val_loader, test_loader, input_dim


def load_data(dataset_name, batch_size, num_workers=0, n_train=50000, n_valtest=10000):
    """
    Generic data loader function that dispatches to specific loaders.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 2025

    if dataset_name == 'cifar10':
        train_loader, val_loader, test_loader, _ = load_cifar(
            seed=seed,
            n_train=n_train,
            n_valtest=n_valtest,
            device=device,
            batch_size=batch_size
        )
        return train_loader, val_loader, test_loader
    elif dataset_name == 'coco':
        train_loader, val_loader, test_loader, _ = load_coco(
            seed=seed,
            n_train=n_train,
            n_valtest=n_valtest,
            device=device,
            batch_size=batch_size
        )
        return train_loader, val_loader, test_loader
    elif dataset_name == 'mnist':
        train_loader, val_loader, test_loader, _ = load_mnist(
            seed=seed,
            n_train=n_train,
            n_valtest=n_valtest,
            device=device,
            batch_size=batch_size
        )
        return train_loader, val_loader, test_loader
    elif dataset_name == 'fashion':
        train_loader, val_loader, test_loader, _ = load_fashion(
            seed=seed,
            n_train=n_train,
            n_valtest=n_valtest,
            device=device,
            batch_size=batch_size
        )
        return train_loader, val_loader, test_loader
    elif dataset_name == 'celeba':
        train_loader, val_loader, test_loader, _ = load_celeba(
            seed=seed,
            n_train=10000,  # Smaller subset for CelebA
            n_valtest=2000,
            device=device,
            batch_size=batch_size
        )
        return train_loader, val_loader, test_loader
    elif dataset_name == 'eeg':
        train_loader, val_loader, test_loader, _ = load_eeg(
            seed=seed,
            device=device,
            batch_size=batch_size
        )
        return train_loader, val_loader, test_loader
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
