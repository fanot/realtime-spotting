import os
import sys
import json
import time
import torch
import librosa
import requests
import threading
import numpy as np
import torch.nn as nn
import soundfile as sf
import torch.optim as optim
from src.logger import logger
import torch.nn.functional as F
from torchmetrics import Accuracy
from torch.utils.data import DataLoader
from src.models.matchboxnet import MatchboxNet
from sklearn.model_selection import train_test_split
from src.KeywordSpottingDataset import KeywordSpottingDataset

# Define constants
SAMPLE_RATE = 16000  # The sample rate of the audio data
HOP_LENGTH = 0.3  # The amount of second on which window will move each iteration.
WINDOW_SIZE = 1  # The amount of seconds in every frame
OUTPUT_FOLDER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), './../outputs')  # The path for the outpu directory
OUTPUT_RADIO_FILE_PATH = os.path.join(OUTPUT_FOLDER_PATH, 'radio.txt')  # The output path for the radio file
TEMP_AUDIO_FILE_PATH = os.path.join(OUTPUT_FOLDER_PATH, 'temp.mp3')  # The temporary file path for the audio file

logger.info(f"Constants: SAMPLE_RATE: {SAMPLE_RATE}, HOP_LENGTH: {HOP_LENGTH}, WINDOW_SIZE: {WINDOW_SIZE},"
            f" OUTPUT_FOLDER_PATH: {OUTPUT_FOLDER_PATH}, OUTPUT_RADIO_FILE_PATH: {OUTPUT_RADIO_FILE_PATH},"
            f" TEMP_AUDIO_FILE_PATH: {TEMP_AUDIO_FILE_PATH}")


class KeywordSpotter:
    """
    A class for detecting wake words in audio files.

    Parameters:
        model_path (str): The path to a pre-trained model.

    Attributes:
        SAMPLE_RATE (int): The sample rate of the audio files.
        hop_length (float): The hop length in seconds.
        window_size (float): The window length in seconds.
        model (MatchboxNet): The pre-trained model used for prediction.
        training_data (None or dict): The training data used for the pre-trained model.

    Methods:
        check_folders(): Creates the output folder if it doesn't exist.
        detect_wake_words(audio_file_path): Detects wake words in an audio file and returns a list of times where the wake word was detected.

    Example:
        spotter = KeywordSpotter('models/matchboxnet.pth')
        res = spotter.detect_wake_words('audio_files/sample.wav')
        print(res)
    """

    def __init__(self, model_path):
        logger.info(f"started initializing KeywordSpotter object, model_path={model_path}")
        """
        Initializes the KeywordSpotter class with the necessary parameters and loads the pre-trained model.

        Parameters:
            model_path (str): The path to a pre-trained model.
        """
        self.SAMPLE_RATE = SAMPLE_RATE  # The sample rate of the audio files.
        self.hop_length = HOP_LENGTH  # The hop length in seconds.
        self.window_size = WINDOW_SIZE  # The window length in seconds.
        self.model = MatchboxNet(B=3, R=2, C=64, bins=64, NUM_CLASSES=2)
        # self.model.load_state_dict(torch.load(model_path))
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.float().to("cpu")
        self.training_data = None
        self.radio_listening_timeout = 4 * 60 * 60   # in seconds, 4 hours
        logger.info(f"KeywordSpotter object initialized, model_path={model_path}")

    def check_folders(self):
        logger.info(f"started checking folders")
        """
        Creates the output folder if it doesn't exist.
        """
        if not os.path.exists(OUTPUT_FOLDER_PATH):
            os.mkdir(OUTPUT_FOLDER_PATH)
            logger.info(f"Output folder created: {OUTPUT_FOLDER_PATH}")
        logger.info(f"Folders checked")


    def detect_wake_words(self, audio_file_path):
        logger.info(f"started detecting words, audio_file_path={audio_file_path}")
        """
       Detects wake words in an audio file and returns a list of times where the wake word was detected.

       Parameters:
           audio_file_path (str): The path to the audio file.

       Returns:
           list: A list of times (in seconds) where the wake word was detected.

       """
        res = []
        hop_length = int(self.hop_length * self.SAMPLE_RATE)  # 0.5 second
        output_file = os.path.join(OUTPUT_FOLDER_PATH, os.path.basename(audio_file_path).replace(".wav", ".txt"))
        with open(output_file, 'w') as f:
            pass  # clear the file

        # load file
        y, sr = librosa.load(audio_file_path, sr=self.SAMPLE_RATE)
        logger.info("audio file successfully loaded")

        # add paddings if it's needed
        n_pad = self.window_size - (len(y) % self.window_size)
        y_padded = np.concatenate((y, np.zeros(n_pad)))
        sec = self.hop_length
        frames = int(y.shape[0] / hop_length)
        self.model.eval()
        with torch.no_grad():
            for i in range(frames):
                window = y_padded[int(sec * self.SAMPLE_RATE): int(
                    sec * self.SAMPLE_RATE + (self.SAMPLE_RATE * self.window_size))]
                tensor = self.convert_audio_to_tensor(window)
                prediction = self.model(tensor)
                probs = F.softmax(prediction, dim=1)
                preds = torch.argmax(probs, dim=1).item()
                print(f"sec {sec}:", preds)
                threshold = 0.5
                if preds > threshold:
                    res.append(sec)
                    with open(output_file, 'a') as f:
                        f.write(
                            os.path.join(OUTPUT_FOLDER_PATH, f'{os.path.basename(audio_file_path)}:{sec + 1:.0f}\n'))
                    cut_name = (os.path.join(OUTPUT_FOLDER_PATH,
                                             os.path.basename(audio_file_path).replace(".wav", f"_{sec + 1:.0f}.wav")))
                    sf.write(cut_name, y_padded[int(sec * SAMPLE_RATE) + (1 * SAMPLE_RATE): int(
                        sec * SAMPLE_RATE + (SAMPLE_RATE * 2.5))], samplerate=SAMPLE_RATE)
                sec += self.hop_length
        logger.info("wake up words successfully detected")
        return res

    def convert_audio_to_tensor(self, audio):
        """Converts an audio signal to an MFCC tensor.

        Args:
            audio (ndarray): Input audio signal.

        Returns:
            Tensor: MFCC tensor of shape (1, 64, T) where T is the number of frames.
        """
        audio = self.model.emphasis(audio)
        mfcc = librosa.feature.mfcc(y=audio, sr=self.SAMPLE_RATE, n_mfcc=64)
        inputs = self.model.padding(torch.from_numpy(mfcc.reshape(1, 64, -1)), 128)
        tensor = inputs.reshape(1, 64, -1).float()
        return tensor

    def process_radio_stream(self, radio_url):
        logger.info(f"started listening to radio stream, radio_url={radio_url}")
        """Processes a radio stream for keyword detection.

        Args:
            radio_url (str): URL of the radio stream.

        Returns:
            None
        """
        connection_attempts = 10000000000000000000  # number of attempts to connect to radio
        wait_time = 10  # time in seconds for next connection try
        for i in range(connection_attempts):
            logger.info(f"try {i+1}/{connection_attempts} to connect to: {radio_url}")
            try:
                # set up timeout handler
                timer = threading.Timer(self.radio_listening_timeout, self.timeout_handler)
                timer.start()
                logger.info(f"started timer")

                output_file = OUTPUT_RADIO_FILE_PATH
                with open(output_file, 'w') as f:
                    f.write(f'Listening to {radio_url}:\n')

                start_flag = True
                while not start_flag:
                    try:
                        audio = requests.get(radio_url, stream=True)
                        audio.raise_for_status()
                        start_flag = True
                        logger.info("Connected to radio stream")
                    except requests.exceptions.RequestException as e:
                        logger.error(f"Failed to connect, retrying in {wait_time} seconds: {e}")
                        time.sleep(wait_time)

                audio = requests.get(radio_url, stream=True)
                audio.raise_for_status()

                streamed_audio = np.array([])
                timing = 0
                next_allowed_timing = 0
                keyword_timing = 0
                keyword = False

                self.model.eval()
                for chunk in audio.iter_content(chunk_size=8192):
                    if chunk:
                        audio_file = open(TEMP_AUDIO_FILE_PATH, 'wb+')
                        audio_file.write(chunk)
                        audio_file.close()
                        try:
                            my_signal, sample_rate = librosa.load(TEMP_AUDIO_FILE_PATH, sr=self.SAMPLE_RATE)
                        except Exception as e:
                            logger.info(f"error while loading temp file", e)
                            continue

                        timing += my_signal.shape[0] / sample_rate
                        streamed_audio = np.concatenate([streamed_audio, my_signal])

                        # get 1st second then start to predict
                        while next_allowed_timing + 1 < timing:
                            print(f"listned {next_allowed_timing + 1} seconds", end=": ")
                            frame = streamed_audio[int((next_allowed_timing) * sample_rate):int((next_allowed_timing + 1) * sample_rate)]
                            frame = self.convert_audio_to_tensor(frame)
                            with torch.no_grad():
                                prediction = self.model(frame)
                                probs = F.softmax(prediction, dim=1)
                                pred = torch.argmax(probs, dim=1).item()
                            threshold = 0.5
                            if pred > threshold:
                                print(f"found on {keyword_timing} second; pred: {pred}")
                                keyword = True
                                keyword_timing = next_allowed_timing
                            else:
                                print(f"nothing found; pred: {pred}")

                            # move window for hop_length to predict next frame
                            next_allowed_timing += self.hop_length
                        if keyword and (keyword_timing + 3 < timing):
                            print(f"DETECTED!!!")
                            logger.info(f"DETECTED ON {keyword_timing} second")
                            with open(output_file, 'a') as f:
                                f.write(f'{radio_url}:{int(next_allowed_timing + 1):.0f}\n')

                            cut_name = os.path.join(OUTPUT_FOLDER_PATH, f'radio_{int(keyword_timing)}.wav')
                            keyword_frame = streamed_audio[
                                           int((keyword_timing - 1) * sample_rate):int((keyword_timing + 3) * sample_rate)]
                            sf.write(cut_name, keyword_frame, samplerate=self.SAMPLE_RATE)
                            logger.info(f"saved detected frame, cut_name={cut_name}")
                            keyword = False
            except KeyboardInterrupt:
                self.remove_temp_files()
                logger.info(f"user cancelled process")
                os._exit(0)
            except Exception as e:
                logger.warn(e)
                time.sleep(wait_time)

    def remove_temp_files(self):
        if os.path.exists(TEMP_AUDIO_FILE_PATH):
            os.remove(TEMP_AUDIO_FILE_PATH)
        logger.info(f"temp files removed on: {TEMP_AUDIO_FILE_PATH}")

    def timeout_handler(self):
        logger.error("Process timed out")
        self.remove_temp_files()
        os._exit(0)

    def choose_device(self):
        """
        Method to choose the device for training the model.

        Returns:
            device (str): "cuda" if a GPU is available, otherwise "cpu".
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"chose device: {device}")
        return device

    def load_data(self, data_path):
        logger.info(f"started loading data, data_path={data_path}")
        """
        Method to load the training data from a JSON file and store it in the KeywordSpotter object.

        Args:
            data_path (str): path to the JSON file containing the training data.

        Returns:
            None
        """
        f = open(data_path)
        data = json.load(f)
        for old_key in list(data.keys()):
            new_key = data_path.replace(
                "data.json", old_key)
            data[new_key] = data.pop(old_key)
        self.training_data = data
        logger.info(f"data successfully loaded")

    def reload_model(self, model_path=None):
        logger.info(f"started model reloading, model_path={model_path}")
        """
        Method to reload the MatchboxNet model for training.

        Args:
            model_path (str): path to the saved model file (default: None).

        Returns:
            None
        """
        self.model = MatchboxNet(B=3, R=2, C=64, bins=64, NUM_CLASSES=2)
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path))
        self.model.float().to("cpu")
        logger.info(f"model successfully reloaded")

    def prepare(self, data_path, batch_size, test_size=0.2):
        logger.info(f"started preparing data: data_path={data_path}; batch_size={batch_size}; test_size={test_size}")
        """
       Method to prepare the training and validation data for training the model.

       Args:
           data_path (str): path to the JSON file containing the training data.
           batch_size (int): batch size for training the model.
           test_size (float): fraction of the data to use for validation (default: 0.2).

       Returns:
           train_dataloader (DataLoader): dataloader for the training data.
           val_dataloader (DataLoader): dataloader for the validation data.
           criterion (CrossEntropyLoss): loss function for training the model.
           optimizer (Adam): optimizer for training the model.
           device (str): "cuda" if a GPU is available, otherwise "cpu".
       """
        self.load_data(data_path)
        device = self.choose_device()

        # prepare data
        file_paths = [path for path in self.training_data.keys()]
        labels = [label for label in self.training_data.values()]
        X_train_paths, X_val_paths, y_train, y_val = train_test_split(file_paths, labels, test_size=test_size,
                                                                      random_state=42, shuffle=True, stratify=labels)
        logger.info("data splitted")

        # create train and validation datasets
        train_dataset = KeywordSpottingDataset(X_train_paths, y_train)
        val_dataset = KeywordSpottingDataset(X_val_paths, y_val)
        logger.info("datasets created")

        # create train and validation dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        logger.info("dataloaders created")

        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        logger.info("objects successfully prepared")
        return train_dataloader, val_dataloader, criterion, optimizer, device

    def run_epochs(self, model, loader, criterion, optimizer, device):
        logger.info(f"started running new epochs: model={type(model)}; loader={loader}; criterion={criterion}; optimizer={optimizer}; device={device}")
        """
        Trains the given model for one epoch on the given data loader using the given criterion and optimizer.

        Args:
            model (nn.Module): The model to train.
            loader (DataLoader): The data loader to use for training.
            criterion (nn.Module): The loss criterion to use for training.
            optimizer (optim.Optimizer): The optimizer to use for training.
            device (str): The device to use for training.

        Returns:
            tuple: A tuple containing the average loss and accuracy achieved during the epoch.
        """
        model.to(device)
        model.train()
        running_loss = 0.0
        acc = 0
        count = 0
        for i, (inputs, targets) in enumerate(loader):
            inputs = inputs.float().to(device)
            targets = targets.type(torch.LongTensor).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            running_acc = Accuracy(num_classes=2, compute_on_step=False, dist_sync_on_step=False, task='binary')(
                predicted.cpu(), targets.cpu())
            acc += running_acc
            count = i
            print(f"training running batch accuracy: {running_acc}")
        accuracy = acc / (count + 1)
        logger.info(f"all epochs successfully completed")
        return running_loss / len(loader), accuracy

    def validate(self, model, loader, criterion, device):
        logger.info(
            f"starting validation: model={type(model)}; loader={loader}; criterion={criterion}; device={device}")
        """
        Runs validation on the given model using the given data loader and criterion.

        Args:
            model (nn.Module): The model to validate.
            loader (DataLoader): The data loader to use for validation.
            criterion (nn.Module): The loss criterion to use for validation.
            device (str): The device to use for validation.

        Returns:
            tuple: A tuple containing the average loss and accuracy achieved during validation.
        """
        model.eval()
        model.to(device)
        running_loss = 0.0
        acc = 0
        count = 0
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(loader):
                inputs = inputs.float().to(device)
                targets = targets.type(torch.LongTensor).to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                running_acc = Accuracy(num_classes=2, compute_on_step=False, dist_sync_on_step=False, task='binary')(
                    predicted.cpu(), targets.cpu())
                print('training running accuracy', running_acc)
                acc += running_acc
                count = i
            accuracy = acc / (count + 1)
        logger.info(f"validation successfully completed")
        return running_loss / len(loader), accuracy

    def train(self, data_path, batch_size, n_epochs, test_size):
        logger.info(f"started training: data_path={data_path}; batch_size={batch_size}; n_epochs={n_epochs}; test_size={test_size}")
        """
        Trains the MatchboxNet model on the provided data.

        Args:
            data_path (str): Path to the JSON data file.
            batch_size (int): The batch size to be used for training.
            n_epochs (int): Number of epochs to train the model.
            test_size (float): The fraction of data to be used for validation.

        Returns:
            None

        """
        train_dataloader, val_dataloader, criterion, optimizer, device = self.prepare(data_path, batch_size, test_size)
        self.reload_model()
        for epoch in range(n_epochs):
            train_loss, train_acc = self.run_epochs(self.model, train_dataloader, criterion, optimizer, device)
            val_loss, val_acc = self.validate(self.model, val_dataloader, criterion, device)
            logger.info(f"training epoch {epoch + 1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")
        model_path = f'./src/models/store/MatchboxNet_{n_epochs}_epochs.pth'
        torch.save(self.model.state_dict(), model_path)
        self.reload_model(model_path)
        logger.info(f"model trained, saved to: {model_path}")

    def evaluate(self, data_path, batch_size):
        logger.info(
            f"started evaluating: data_path={data_path}; batch_size={batch_size}")
        """
        Evaluates the MatchboxNet model on the provided data.

        Args:
           data_path (str): Path to the JSON data file.
           batch_size (int): The batch size to be used for evaluation.

        Returns:
           None

        """
        _, val_dataloader, criterion, optimizer, device = self.prepare(data_path, batch_size)
        val_loss, val_acc = self.validate(self.model, val_dataloader, criterion, device)
        logger.info(f"Evaluating results: Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")
        logger.info("model successfully evaluated")