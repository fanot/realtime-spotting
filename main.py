import os
import fire
import src.logger
from src.logger import logger
from src.KeywordSpotter import KeywordSpotter

TRAINING_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), './dataset/data.json')  # Path to the
# training data file containing audio and labels in JSON format
BATCH_SIZE = 16  # Number of samples in each batch of data during training
N_EPOCHS = 5  # Number of times the entire training dataset is passed through the model during training
TEST_SIZE = 0.3  # The proportion of the training data to be used as the validation set during training

logger.info(f"Constants: TRAINING_DATA: {TRAINING_DATA}, BATCH_SIZE: {BATCH_SIZE}, N_EPOCHS: {N_EPOCHS},"
            f" TEST_SIZE: {TEST_SIZE}")


class CliWrapper(object):
    """
    This class is a CLI (Command Line Interface) wrapper for the `KeywordSpotter` class.
    It allows the user to interact with the `KeywordSpotter` class using command line arguments.
    """

    def __init__(self):
        """
        Initializes a new `CliWrapper` instance.

        Args:
            None

        Returns:
            None
        """
        self.spotter = None
        self.default_model = './src/models/store/MatchboxNet_1_sec.pth'
        logger.info("CliWrapper object initialized")

    def __load_model(self, model_path=None):
        logger.info(f"started loading model on, model_path={model_path}")
        """
        Loads a pre-trained model from a given path or the default path.

        Args:
            model_path (str, optional): The path to the pre-trained model. Defaults to `None`.

        Returns:
            None
        """
        if model_path is None:
            model_path = self.default_model
        self.spotter = KeywordSpotter(model_path)
        self.spotter.check_folders()
        logger.info(f"Model loaded successfully, model_path={model_path}")

    def find(self, audio='./training_data/data/thanos_message.wav', model_path=None):
        """
        Finds wake words in an audio file using a pre-trained model.

        Args:
           audio (str, optional): The path to the audio file to process. Defaults to `'./training_data/data/thanos_message.wav'`.
           model_path (str, optional): The path to the pre-trained model. Defaults to `None`.

        Returns:
           None
        """
        self.__load_model(model_path)
        logger.info(f"Started detecting wake words on audio={audio}; model_path={model_path}")
        results = self.spotter.detect_wake_words(os.path.join(os.path.dirname(os.path.abspath(__file__)), audio))
        logger.info(f'Keywords detected on {", ".join([str(x) for x in results])} seconds')
        logger.info(f"The method is successfully finished")

    def listen(self, radio='https://radio.kotah.ru/soundcheck', model_path=None):
        """
        Listens to a live radio stream and detects wake words using a pre-trained model.

        Args:
            radio (str, optional): The URL of the radio stream to listen to.
            model_path (str, optional): The path to the pre-trained model. Defaults to `None`.

        Returns:
            None
        """
        self.__load_model(model_path)
        logger.info(f"Started listening to radio stream: radio={radio}; model_path={model_path}")
        self.spotter.process_radio_stream(radio)
        logger.info(f"The method is successfully finished")

    def listen_real(self, radio='https://radio.kotah.ru/exam', model_path=None):
        """
        Temp method only for coursework. The same method that `listen` but only for one specified radio station
        Listens to a live radio stream and detects wake words using a pre-trained model.

        Args:
           radio (str, optional): The URL of the radio stream to listen to.
           model_path (str, optional): The path to the pre-trained model. Defaults to `None`.

        Returns:
           None
        """
        self.__load_model(model_path)
        logger.info(f"Started listening to radio stream: radio={radio}; model_path={model_path}")
        self.spotter.process_radio_stream(radio)
        logger.info(f"The method is successfully finished")

    def train(self, data_path=TRAINING_DATA, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, test_size=TEST_SIZE):
        """
        Trains the KeywordSpotter model using the provided training data.

        Parameters:
            data_path (str): The file path of the JSON file containing the training data.
                Defaults to the constant value TRAINING_DATA.
            batch_size (int): The batch size to use during training. Defaults to the constant value BATCH_SIZE.
            n_epochs (int): The number of epochs to train for. Defaults to the constant value N_EPOCHS.
            test_size (float): The fraction of the data to use for testing. Defaults to the constant value TEST_SIZE.

        Returns:
            None
        """
        self.__load_model()
        logger.info(f"Started training: data_path={data_path}; batch_size={batch_size}; n_epochs={n_epochs}' test_size={test_size}")
        self.spotter.train(data_path, batch_size, n_epochs, test_size)
        logger.info(f"The method is successfully finished")

    def evaluate(self, data_path=TRAINING_DATA, batch_size=BATCH_SIZE):
        """
        Evaluates the performance of the KeywordSpotter model using the provided evaluation data.

        Parameters:
            data_path (str): The file path of the JSON file containing the evaluation data.
                Defaults to the constant value TRAINING_DATA.
            batch_size (int): The batch size to use during evaluation. Defaults to the constant value BATCH_SIZE.

        Returns:
            None
        """
        self.__load_model()
        logger.info(f"Started evaluating: data_path={data_path}; batch_size={batch_size}")
        self.spotter.evaluate(data_path, batch_size)
        logger.info(f"The method is successfully finished")


if __name__ == "__main__":
    """
    This block of code checks if the script is being run as the main program, and if it is,
    it creates an instance of the CliWrapper class and uses the Google's python-fire library
    to enable CLI commands.
    """
    fire.Fire(CliWrapper)
