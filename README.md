# EPUB To MP3
Use text-to-speech synthesis to create an MP3 audiobook of an eBook (in .epub format).

## Development Environment
Developed with Python 3 on Ubuntu 18.04, and uses Google Cloud Speech-to-Text API.

## Setup

#### MP3 Codecs
`ffmpeg` is used by the `pydub` library to cocatenate MP3s together. This is needed
because an audiobook will need to be separated into chunks, sent over to the Google
Cloud Platfrom Text-to-Speech API, and the individual chunk's MP3s will be concatenated.

```sh
$ sudo apt update
$ sudo apt install ffmpeg 
```

#### GPU Support for offline TTS
Highly recomended if you plan to do the speech synthesis on your local machine, as this is _quite_ slow without GPU acceleration.

For offline models, if you have a GPU and are on WSL, [follow these instructions](https://docs.nvidia.com/cuda/wsl-user-guide/index.html). You'll want to run
 ```sh
$ sudo apt install nvidia-cuda-toolkit
$ sudo apt-get install -y cuda-toolkit-10-1
``` 
after installing the drivers on windows, which is a little unclear from the instructions. Check that everything is intalled by running ```nvcc --version``` in the linux terminal, you should have 10.1.

If not, you may also need to follow the vanilla linux isntructions below to add cuda 10.1 to your PATH.

If you are on on plain old linux, [follow these instructions](https://medium.com/@exesse/cuda-10-1-installation-on-ubuntu-18-04-lts-d04f89287130)


#### Python
## Online TTS
Set up virutalenv and install libraries for:
* Google Cloud Text to Speech API
* ePub parsing

```sh
$ virtualenv venv
$ source venv\bin\activate
$ pip3 install -r requirements_online_tts.txt
```
## Offline TTS
Set up virutalenv and install libraries for:
* TensorFlow
* ePub parsing

```sh
$ virtualenv venv
$ source venv/bin/activate
$ pip3 install -r requirements_local_tts.txt
```
### Offline model downloads
**Option 1: Download pre-zipped models**

Download [the zip file](https://drive.google.com/file/d/1fcc8Rdl9WemkwFoiLg4p22vilwd5h8KV/view?usp=sharing) and extract to this directory (i.e. the root of the repo).

**Option 2: Download models from source**

Otherwise you can download models to the ```model_files``` folder from their respective [sub-folders in the repo](https://github.com/TensorSpeech/TensorFlowTTS/tree/master/examples). For each The necessary files are the configuration.yaml file and the generator.h5 file

### Google Cloud Platform
To use online recognition, set up a Google Cloud Platform account, following the instructions here: 
https://cloud.google.com/text-to-speech/docs/libraries#client-libraries-install-python

You'll get a JSON file with your Google Cloud Platform credentials. Set it in environment variables.

For example:
```sh
$ export GOOGLE_APPLICATION_CREDENTIALS=~/dev/EPUB-to-MP3/GCP-credentials.json
```

## Usage
```sh
$ python3 convert.py --file="book.epub"
```

## Offline usage
```sh
$ python3 convert.py --file="book.epub" --local
```