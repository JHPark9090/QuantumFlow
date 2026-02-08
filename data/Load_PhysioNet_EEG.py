import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import mne
from sklearn.model_selection import train_test_split

print('MNE Version :', mne.__version__)

def load_eeg_ts_revised(seed, device, batch_size, sampling_freq, sample_size):
    """
    Loads and preprocesses the PhysioNet EEG Motor Imagery dataset for a specified number of subjects.

    Args:
        seed (int): Random seed for reproducibility.
        device (torch.device): The device to move the tensors to.
        batch_size (int): Number of samples per batch.
        sampling_freq (int): The target sampling frequency to resample the data to.
        sample_size (int): The number of subjects to load data from (1 to 109).

    Returns:
        tuple: A tuple containing (train_loader, test_loader, input_dim).
               - train_loader: DataLoader for the training set.
               - val_loader: DataLoader for the validation set.
               - test_loader: DataLoader for the test set.
               - input_dim: The shape of the input data (n_trials, n_channels, n_timesteps).
    """
    
    # --- Step 1: Split Subject IDs into Train, Validation, and Test Sets ---
    N_SUBJECT = sample_size
    subject_ids = np.arange(1, N_SUBJECT + 1)
    
    # Split subjects: ~70% train, ~15% validation, ~15% test
    train_subjects, temp_subjects = train_test_split(
        subject_ids,
        test_size=0.3, # 30% for validation and test combined
        random_state=seed
    )
    
    val_subjects, test_subjects = train_test_split(
        temp_subjects,
        test_size=0.5, # Split the 30% evenly into 15% validation and 15% test
        random_state=seed
    )
    
    print(f"Subjects in Training Set: {len(train_subjects)}")
    print(f"Subjects in Validation Set: {len(val_subjects)}")
    print(f"Subjects in Test Set: {len(test_subjects)}")

    # --- Step 2: Define a Helper Function to Load Data for a Given List of Subjects ---
    def _load_and_process_subjects(subject_list, sfreq):
        IMAGINE_OPEN_CLOSE_LEFT_RIGHT_FIST = [4, 8, 12]

        # Absolute path to PhysioNet EEG data (avoids interactive prompt in batch jobs)
        DATA_PATH = "/pscratch/sd/j/junghoon/PhysioNet_EEG"

        # Load file paths for the specified subjects
        physionet_paths = [
            mne.datasets.eegbci.load_data(
                subjects=subj_id,
                runs=IMAGINE_OPEN_CLOSE_LEFT_RIGHT_FIST,
                path=DATA_PATH,
                update_path=False,  # Avoid interactive prompt
            ) for subj_id in subject_list
        ]
        physionet_paths = np.concatenate(physionet_paths)

        parts = []
        for path in physionet_paths:
            raw = mne.io.read_raw_edf(
                path, preload=True, stim_channel='auto', verbose='WARNING'
            )
            raw.resample(sfreq, npad="auto")
            parts.append(raw)
        
        # Concatenate all runs FOR THIS SPLIT into one raw object
        raw = mne.concatenate_raws(parts)

        # Epoch the data
        events, _ = mne.events_from_annotations(raw)
        eeg_channel_inds = mne.pick_types(
            raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads'
        )
        epoched = mne.Epochs(
            raw, events, dict(left=2, right=3), tmin=1, tmax=4.1,
            proj=False, picks=eeg_channel_inds, baseline=None, preload=True
        )

        # Convert to NumPy arrays
        X = (epoched.get_data() * 1e3).astype(np.float32)
        y = (epoched.events[:, 2] - 2).astype(np.int64)
        
        return X, y

    # --- Step 3: Load Data Separately for Each Subject Group ---
    X_train, y_train = _load_and_process_subjects(train_subjects, sampling_freq)
    X_val, y_val = _load_and_process_subjects(val_subjects, sampling_freq)
    X_test, y_test = _load_and_process_subjects(test_subjects, sampling_freq)
    
    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    print(f"Test set shape: {X_test.shape}")

    # --- Step 4: Create PyTorch Datasets and DataLoaders ---
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32).to(device),
                                  torch.tensor(y_train, dtype=torch.float32).to(device))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32).to(device),
                                torch.tensor(y_val, dtype=torch.float32).to(device))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32).to(device),
                                 torch.tensor(y_test, dtype=torch.float32).to(device))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    input_dim = X_train.shape
    
    return train_loader, val_loader, test_loader, input_dim