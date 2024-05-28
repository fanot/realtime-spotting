import numpy as np
import librosa
import torch
from torch.utils.data import Dataset
from src.logger import logger

# Define constants
TRAINING_SAMPLE_RATE = 16000  # Sampling rate used during training.
logger.info(f"Constants: TRAINING_SAMPLE_RATE: {TRAINING_SAMPLE_RATE}")


class KeywordSpottingDataset(Dataset):
    """
    A PyTorch Dataset class for loading audio files and labels for keyword spotting.
    """
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        """
        Returns:
            int: The number of audio files in the dataset.
        """
        return len(self.file_paths) - 1

    def emphasis(self, audio, pre_emphasis=0.97):
        """
        Applies pre-emphasis filter to the audio.

        Args:
            audio (np.ndarray): Audio data.
            pre_emphasis (float, optional): Pre-emphasis coefficient. Defaults to 0.97.

        Returns:
            np.ndarray: Audio data with pre-emphasis filter applied.
        """
        audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
        return audio

    def crop_audio(self, audio, sr=16000):
        """
        Replaces a random segment of the audio with silence.

        Args:
            audio (np.ndarray): Audio data.
            sr (int, optional): Sampling rate. Defaults to 16000.

        Returns:
            np.ndarray: Augmented audio data with a random segment replaced with silence.
        """
        # Get length of audio in samples
        audio_len = audio.shape[0]

        silence_length = np.random.uniform(0.3, 0.5)

        # Calculate max number of samples to replace with silence
        silence_len = int(silence_length * sr)

        # randomly choose where to crop audio
        in_start = np.random.choice([True, False])
        if in_start:
            start_idx = 0
            end_idx = start_idx + silence_len
        else:
            start_idx = audio_len - silence_len - 1
            end_idx = audio_len - 1

        # Replace audio segment with silence
        augmented_audio = audio.copy()
        augmented_audio[start_idx:end_idx] = 0.0

        return augmented_audio

    def add_noise(self, audio, noise_level=0.01):
        """
        Add Gaussian noise to the audio signal.

        Parameters:
            audio (np.ndarray): Audio signal.
            noise_level (float): Standard deviation of the Gaussian noise.

        Returns:
            np.ndarray: Noisy audio signal.
        """
        noise = np.random.normal(scale=noise_level, size=len(audio))
        return audio + noise

    def pitch_shift(self, audio, sr=16000):
        """
        Shift the pitch of the audio signal by a random number of semitones.

        Parameters:
            audio (np.ndarray): Audio signal.
            sr (int): Sampling rate of the audio signal.

        Returns:
            np.ndarray: Pitch-shifted audio signal.
        """
        steps = [-3, -2, -1, 1, 2, 3]
        choice = np.random.choice(steps, 1)
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=choice)

    def augment_audio(self, audio):
        """
        Apply random augmentations to the audio signal.

        Parameters:
            audio (np.ndarray): Audio signal.

        Returns:
            np.ndarray: Augmented audio signal.
        """
        # Randomly apply one or more augmentation methods
        methods = []
        if np.random.random() < 0.5:
            methods.append(self.crop_audio)
        if np.random.random() < 0.5:
            methods.append(self.add_noise)
        if np.random.random() < 0.5:
            methods.append(self.pitch_shift)

        # Apply selected augmentation methods to audio
        for method in methods:
            audio = method(audio)

        return audio

    def get_log_mel_spectrogram(self, audio):
        """
        Compute the log-mel spectrogram of the audio signal.

        Parameters:
            audio (np.ndarray): Audio signal.

        Returns:
            np.ndarray: Log-mel spectrogram.
        """
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=TRAINING_SAMPLE_RATE, n_fft=4096, hop_length=512, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_db = mel_spec_db.astype(np.float32)
        return mel_spec_db


    def normalize_spectrogram(self, mel_spec_db):
        """
        Normalize a mel spectrogram by scaling to a range between 0 and 1.

        Args:
            mel_spec_db (numpy.ndarray): The mel spectrogram to be normalized.

        Returns:
            numpy.ndarray: The normalized mel spectrogram.
        """
        mel_spec_db = librosa.util.normalize(mel_spec_db)
        return mel_spec_db

    def get_mfcc(self, audio):
        """
        Compute the Mel-frequency cepstral coefficients (MFCCs) from an audio signal.

        Args:
            audio (numpy.ndarray): The audio signal.

        Returns:
            numpy.ndarray: The MFCCs.
        """
        mfcc = librosa.feature.mfcc(y=audio, sr=TRAINING_SAMPLE_RATE, n_mfcc=40, n_fft=4096, hop_length=512)
        mfcc = mfcc.astype(np.float32)
        return mfcc

    def pad_mfccs(self, mfccs):
        """
        Pad MFCCs with zeros to match a predefined shape.

        Args:
            mfccs (numpy.ndarray): The MFCCs to be padded.

        Returns:
            numpy.ndarray: The padded MFCCs.
        """
        pad_width = ((0, 128 - mfccs.shape[0]), (0, 0))
        padded_mfccs = np.pad(mfccs, pad_width, mode='constant')
        return padded_mfccs

    def merge_spec_and_mfcc(self, mel_spec_db, mfccs):
        """
        Merge a mel spectrogram and MFCCs into a single tensor.

        Args:
            mel_spec_db (numpy.ndarray): The mel spectrogram.
            mfccs (numpy.ndarray): The MFCCs.

        Returns:
            torch.Tensor: The merged tensor.
        """
        spectrograms = torch.stack([torch.from_numpy(mel_spec_db), torch.from_numpy(mfccs)])
        return spectrograms

    def padding(self, batch, seq_len):
        """
        Pad a batch of spectrograms with zeros to match a predefined sequence length.

        Args:
            batch (torch.Tensor): The batch of spectrograms.
            seq_len (int): The desired sequence length.

        Returns:
            torch.Tensor: The padded batch of spectrograms.
        """
        if len(batch[0][0]) < seq_len:
            m = torch.nn.ConstantPad1d((0, seq_len - len(batch[0][0])), 0)
            batch = m(batch)
        return batch

    def __getitem__(self, idx):
        """
        Retrieves the audio file and corresponding label at the given index, preprocesses the audio,
        computes the MFCC features, pads the features to a fixed length, and returns the features
        along with the label.

        Args:
            idx (int): Index of the audio file to retrieve.

        Returns:
            tuple: A tuple containing the MFCC features and corresponding label as PyTorch tensors.
        """
        audio_path = self.file_paths[idx]
        label = self.labels[idx]

        # Load and preprocess audio
        audio, _ = librosa.load(audio_path, sr=TRAINING_SAMPLE_RATE)
        audio = self.emphasis(audio)
        audio = self.augment_audio(audio)

        # Compute MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=TRAINING_SAMPLE_RATE, n_mfcc=64)

        # Pad features to fixed length
        inputs = self.padding(torch.from_numpy(mfcc.reshape(1, 64, -1)), 128)
        tensor = inputs.reshape(64, -1)

        # Convert label to PyTorch tensor
        label = torch.tensor(label, dtype=torch.long)
        label.numpy()
        return tensor, label
