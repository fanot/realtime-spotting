# Sound Processing and Real-Time Keyword Spotting

This project is designed to detect keywords in audio files and live radio streams.
## Models

We utilize the [MatchboxNet](https://arxiv.org/abs/2004.08531) model, an end-to-end neural network tailored for speech command recognition.

## Installation

No installation is necessary. Simply download and unpack the repository files. Note that the datasets for model training are not included in the repository. Download the datasets from [here](https://drive.google.com/file/d/1ONZ8JSa93GXT8f6FAft7q7430lcU3kWl/view?usp=sharing) and unpack them into the project's root folder.

To install the required dependencies, run:
```bash
$ pip install -r requirements.txt
```

## User Manual

The system can be operated using several Command Line Interface (CLI) commands. Below is an explanation of each command and its parameters:

### CLI Commands

1. **Train:**
   Initiates the training of the KeywordSpotter model using specified training data and saves the model to the designated directory. The command accepts parameters such as the path to the training data, batch size, number of epochs, and the test size.

   ```bash
   $ python path/to/main.py train --parameter=value
   ```

   Example:
   ```bash
   $ python path/to/main.py train --data_path='./data/train_data.json' --batch_size=16 --n_epochs=5 --test_size=0.5
   ```

2. **Evaluate:**
   Evaluates the performance of the trained KeywordSpotter model using provided evaluation data. It assesses the model's accuracy and other performance metrics.

   ```bash
   $ python path/to/main.py evaluate --parameter=value
   ```

   Example:
   ```bash
   $ python path/to/main.py evaluate --data_path='./data/eval_data.json' --batch_size=16
   ```

3. **Listen:**
   Listens to a live radio stream and detects wake words. It saves audio clips containing the keywords to an output folder. This command also ends after a predefined time or can be manually terminated.

   ```bash
   $ python path/to/main.py listen --parameter=value
   ```

   Example:
   ```bash
   $ python path/to/main.py listen --radio='https://radio.example.com/stream' --model_path='./models/keyword_spotter_model'
   ```

4. **Find:** 
   Processes an audio file to identify and extract segments following detected keywords. Outputs the cropped audio segments to a specified directory.

   ```bash
   $ python path/to/main.py find --parameter=value
   ```

   Example:
   ```bash
   $ python path/to/main.py find --audio='./audio/sample.wav' --model_path='./models/keyword_spotter_model'
   ```

These commands provide the necessary tools to effectively utilize the KeywordSpotter model across various audio processing scenarios, allowing users to train, evaluate, listen to live streams, and analyze recorded audio efficiently.
