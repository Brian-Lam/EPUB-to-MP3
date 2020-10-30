"""
A wrapper for TTS models from https://github.com/TensorSpeech/TensorFlowTTS

Example code mostly copied and modified from:
https://github.com/TensorSpeech/TensorFlowTTS/blob/master/notebooks/TensorFlowTTS_Tacotron2_with_TFLite.ipynb
"""
#@title Licensed under the Apache License, Version 2.0 (the "License"); { display-mode: "form" }
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import soundfile as sf
import yaml
import tensorflow as tf
import os

# If profiling is necessary:
# import cProfile # https://towardsdatascience.com/finding-performance-bottlenecks-in-python-4372598b7b2c


from tensorflow_tts.processor import LJSpeechProcessor
from tensorflow_tts.processor.ljspeech import LJSPEECH_SYMBOLS


from tensorflow_tts.configs import Tacotron2Config
from tensorflow_tts.configs import MelGANGeneratorConfig
from tensorflow_tts.configs.fastspeech2 import FastSpeech2Config
from tensorflow_tts.configs import MultiBandMelGANGeneratorConfig

from tensorflow_tts.models import TFTacotron2
from tensorflow_tts.models import TFFastSpeech2
from tensorflow_tts.models import TFMelGANGenerator
from tensorflow_tts.models import TFMBMelGANGenerator

from tensorflow_tts.inference import AutoConfig
from tensorflow_tts.inference import TFAutoModel
from tensorflow_tts.inference import AutoProcessor

from collections import OrderedDict

# Filepaths
OUT_TACOTRON_TFLITE_FILE = 'tacotron2.tflite'
OUT_TACOTRON_TFLITE_DIR = './model_files/tacotronv2'
TACOTRON_TFLITE_PATH = './model_files/tacotronv2/tacotron2.tflite'

OUT_FASTSPEECH2_TFLITE_FILE = 'fastspeech2.tflite'
OUT_FASTSPEECH2_TFLITE_DIR = './model_files/fastspeech2'
FASTSPEECH2_TFLITE_PATH = './model_files/fastspeech2/fastspeech2.tflite'

OUT_MELGAN_TFLITE_FILE = 'melgan.tflite'
OUT_MELGAN_TFLITE_DIR = './model_files/melgan'
OUT_MB_MELGAN_TFLITE_FILE = 'mb_melgan.tflite'
OUT_MB_MELGAN_TFLITE_DIR = './model_files/multiband_melgan'
LJSPEECH_PROCESSOR_JSON = './processor/pretrained/ljspeech_mapper.json'

#TODO@allen: handle case where tflite models don't exist! And also hard-coded filepaths r bad
class TTS(object):
    """Initializes a TTS and Vocoder model to consume strings and return .wav files."""

    def __init__(self, tts="fastspeech2", generator="multiband_melgan_generator"):
        CONFIG_MAPPING = OrderedDict(
            [
                ("fastspeech2", (self._load_fastspeech2, self._infer_fastspeech2, FASTSPEECH2_TFLITE_PATH)),
                ("multiband_melgan_generator", (self._load_mb_melgan, OUT_MB_MELGAN_TFLITE_DIR)),
                ("melgan_generator", (self._load_melgan, OUT_MELGAN_TFLITE_DIR)),
                ("tacotron2", (self._load_tacotron, self._infer_tacotron2, TACOTRON_TFLITE_PATH)),
            ]
        )
        try:
            _tts, _inference, _tflite_path = CONFIG_MAPPING[tts]
            _generator, _mel_tflite_path = CONFIG_MAPPING[generator]
        except Exception:
            raise ValueError(
                "Unrecognized tts ({}) or generator ({}). "
                "Supported models are: {}".format(
                    tts, generator, ", ".join(CONFIG_MAPPING.keys())
                )
            )
        # self._tts = _tts() # TTS Model, unused if we use tflite model
        self._generator = _generator() # MelGan Vocoder
        self._inference = _inference # TTS Inference function call

        self._processor = LJSpeechProcessor(None, symbols=LJSPEECH_SYMBOLS)
        self._interpreter = tf.lite.Interpreter(model_path=_tflite_path)
        self._interpreter.allocate_tensors()

        # Get input and output tensors.
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

        config = os.path.join(_mel_tflite_path, 'config.yml')

        with open(config) as f:
            melgan_config = yaml.load(f, Loader=yaml.Loader)
        self._sampling_rate = melgan_config["sampling_rate"]

    def _load_melgan(self, path='./model_files/melgan'):
        # initialize melgan model for vocoding
        config = os.path.join(path, 'config.yml')
        with open(config) as f:
            melgan_config = yaml.load(f, Loader=yaml.Loader)
        melgan_config = MelGANGeneratorConfig(
            **melgan_config["generator_params"])
        melgan = TFMelGANGenerator(
            config=melgan_config, name='melgan_generator')
        melgan._build()
        weights = os.path.join(path, 'generator-1670000.h5')
        melgan.load_weights(weights)
        return melgan

    def _load_mb_melgan(self, path='./model_files/multiband_melgan'):
        # initialize melgan model for vocoding
        config = os.path.join(path, 'config.yml')
        with open(config) as f:
            melgan_config = yaml.load(f, Loader=yaml.Loader)
        melgan_config = MultiBandMelGANGeneratorConfig(
            **melgan_config["generator_params"])
        melgan = TFMBMelGANGenerator(
            config=melgan_config, name='melgan_generator')
        melgan._build()
        weights = os.path.join(path, 'generator-940000.h5')
        melgan.load_weights(weights)
        return melgan

    def _load_fastspeech2(self, path='./model_files/fastspeech2'):
        config = os.path.join(path, 'config.yml')
        with open(config) as f:
            config = yaml.load(f, Loader=yaml.Loader)
        config = FastSpeech2Config(**config["fastspeech_params"])
        fastspeech2 = TFFastSpeech2(config=config, name="fastspeech2v1",
                                    enable_tflite_convertible=True)

        fastspeech2._build()
        weights = os.path.join(path, 'model-150000.h5')
        fastspeech2.load_weights(weights)
        print(fastspeech2.summary())
        return fastspeech2

    def _load_tacotron(self, path=OUT_TACOTRON_TFLITE_DIR):
        # initialize Tacotron2 model.
        config = os.path.join(path, 'config.yml')
        with open(config) as f:
            config = yaml.load(f, Loader=yaml.Loader)
        config = Tacotron2Config(**config["tacotron2_params"])
        tacotron2 = TFTacotron2(config=config, training=False, name="tacotron2v1",
                                enable_tflite_convertible=True)

        # Newly added :
        tacotron2.setup_window(win_front=6, win_back=6)
        tacotron2.setup_maximum_iterations(3000)

        tacotron2._build()
        weights = os.path.join(path, 'model-120000.h5')
        tacotron2.load_weights(weights)
        print(tacotron2.summary())
        return tacotron2

    def _generate_tflite(self, model, out_file, out_dir):
        # Concrete Function
        model_concrete_function = model.inference_tflite.get_concrete_function()
        converter = tf.lite.TFLiteConverter.from_concrete_functions(
            [model_concrete_function]
        )
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                               tf.lite.OpsSet.SELECT_TF_OPS]
        tflite_model = converter.convert()
        out = os.path.join(out_dir, out_file)
        # Save the TF Lite model.
        with open(out, 'wb') as f:
            f.write(tflite_model)
        print('Model size is %f MBs.' % (len(tflite_model) / 1024 / 1024.0))

    def generate_fastspeech2_tflite(self):
        model = self._load_fastspeech2()
        self._generate_tflite(model, OUT_FASTSPEECH2_TFLITE_FILE,
                             OUT_FASTSPEECH2_TFLITE_DIR)

    def generate_tacotron_tflite(self):
        model = self._load_tacotron()
        self._generate_tflite(
            model, OUT_TACOTRON_TFLITE_FILE, OUT_TACOTRON_TFLITE_DIR)

    def generate_melgan_tflite(self):
        model = self._load_tacotron()
        self._generate_tflite(model, OUT_MELGAN_TFLITE_FILE,
                             OUT_MELGAN_TFLITE_DIR)

    def generate_multiband_melgan_tflite(self):
        model = self._load_tacotron()
        self._generate_tflite(
            model, OUT_MB_MELGAN_TFLITE_FILE, OUT_MB_MELGAN_TFLITE_DIR)

    # Prepare input data.

    def _prepare_input_tacotron2(self, input_ids):
        return (tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
                tf.convert_to_tensor([len(input_ids)], tf.int32),
                tf.convert_to_tensor([0], dtype=tf.int32))

    def _prepare_input_fastspeech2(self, input_ids):
        input_ids = tf.expand_dims(
            tf.convert_to_tensor(input_ids, dtype=tf.int32), 0)
        return (input_ids,
                tf.convert_to_tensor([0], tf.int32),
                tf.convert_to_tensor([1.0], dtype=tf.float32),
                tf.convert_to_tensor([1.0], dtype=tf.float32),
                tf.convert_to_tensor([1.0], dtype=tf.float32))

    def _infer_fastspeech2(self, input_text, interpreter, input_details, output_details):
        # NOTE: FOR DEBUGGING
        # for x in input_details:
        #     print(x)
        # for x in output_details:
        #     print(x)
        input_ids = self._processor.text_to_sequence(input_text.lower())
        interpreter.resize_tensor_input(input_details[0]['index'],
                                        [1, len(input_ids)])
        interpreter.resize_tensor_input(input_details[1]['index'],
                                        [1])
        interpreter.resize_tensor_input(input_details[2]['index'],
                                        [1])
        interpreter.resize_tensor_input(input_details[3]['index'],
                                        [1])
        interpreter.resize_tensor_input(input_details[4]['index'],
                                        [1])
        interpreter.allocate_tensors()
        input_data = self._prepare_input_fastspeech2(input_ids)
        for i, detail in enumerate(input_details):
            input_shape = detail['shape']
            interpreter.set_tensor(detail['index'], input_data[i])

        interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        return (interpreter.get_tensor(output_details[0]['index']),
                interpreter.get_tensor(output_details[1]['index']))

    def _infer_tacotron2(self, input_text, interpreter, input_details, output_details):
        input_ids = self._processor.text_to_sequence(input_text.lower())
        # eos.
        input_ids = np.concatenate(
            [input_ids, [len(LJSPEECH_SYMBOLS) - 1]], -1)
        interpreter.resize_tensor_input(input_details[0]['index'],
                                        [1, len(input_ids)])
        interpreter.allocate_tensors()
        input_data = self._prepare_input_tacotron2(input_ids)
        for i, detail in enumerate(input_details):
            # NOTE: FOR DEBUGGING 
            # print(detail)
            input_shape = detail['shape']
            interpreter.set_tensor(detail['index'], input_data[i])

        interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        return (interpreter.get_tensor(output_details[0]['index']),
                interpreter.get_tensor(output_details[1]['index']))
        
    def run_inference(self, input_text, out_file_name):
        _, mel_output_tflite = self._inference(
            input_text, self._interpreter, self._input_details, self._output_details)
        audio_after_tflite = self._generator(mel_output_tflite)[0, :, 0]
        sf.write('{}.wav'.format(out_file_name), audio_after_tflite, self._sampling_rate)

# COLAB FOR INFERENCE FROM TFLITE MODELS:
# https://colab.research.google.com/drive/1HudLLpT9CQdh2k04c06bHUwLubhGTWxA?usp=sharing


# Additional info:
# Input text of 777 words (4k chars including spaces) used too much memory locally,
# chunk to ~350 chars (75 words) max for best quality
# 777 words = 12 mb file
# input_text = "Recent research at Harvard has shown meditating\
# for as little as 8 weeks, can actually increase the grey matter in the \
# parts of the brain responsible for emotional regulation, and learning."