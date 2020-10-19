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
$ sudo apt install ffmpeg
```

#### Python
Set up virutalenv and install libraries for:
* Google Cloud Text to Speech API
* ePub parsing

```sh
$ virtualenv venv
$ source venv\bin\activate
$ pip3 install --upgrade google-api-python-client google-cloud-speech google-cloud-texttospeech EbookLib bs4 xml-cleaner pydub
```

#### Google Cloud Platform
Set up a Google Cloud Platform account, following the instructions here: 
https://cloud.google.com/text-to-speech/docs/libraries#client-libraries-install-python

You'll get a JSON file with your Google Cloud Platform credentials. Set it in environment variables.

For example:
```sh
$ export GOOGLE_APPLICATION_CREDENTIALS=~/dev/EPUB-to-MP3/GCP-credentials.json
```

## Usage
```sh
$ python3 convert.py 
```