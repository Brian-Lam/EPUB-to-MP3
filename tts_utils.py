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

OUT_TACOTRON_TFLITE_FILE = 'tacotron2.tflite'
OUT_TACOTRON_TFLITE_DIR = './model_files/tacotronv2'
OUT_FASTSPEECH2_TFLITE_FILE = 'fastspeech2.tflite'
OUT_FASTSPEECH2_TFLITE_DIR = './model_files/fastspeech2'
LJSPEECH_PROCESSOR_JSON = './processor/pretrained/ljspeech_mapper.json'


def load_melgan(path='./model_files/melgan'):
    # initialize melgan model for vocoding
    config = os.path.join(path, 'config.yml')
    with open(config) as f:
        melgan_config = yaml.load(f, Loader=yaml.Loader)
    melgan_config = MelGANGeneratorConfig(**melgan_config["generator_params"])
    melgan = TFMelGANGenerator(config=melgan_config, name='melgan_generator')
    melgan._build()
    weights = os.path.join(path, 'generator-1670000.h5')
    melgan.load_weights(weights)
    return melgan

def load_mb_melgan(path='./model_files/multiband_melgan'):
    # initialize melgan model for vocoding
    config = os.path.join(path, 'config.yml')
    with open(config) as f:
        melgan_config = yaml.load(f, Loader=yaml.Loader)
    melgan_config = MultiBandMelGANGeneratorConfig(**melgan_config["generator_params"])
    melgan = TFMBMelGANGenerator(config=melgan_config, name='melgan_generator')
    melgan._build()
    weights = os.path.join(path, 'generator-940000.h5')
    melgan.load_weights(weights)
    return melgan


def load_fastspeech2(path='./model_files/fastspeech2'):
    config = os.path.join(path, 'config.yml')
    with open(config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    # config = AutoConfig.from_pretrained(config)
    config = FastSpeech2Config(**config["fastspeech_params"])
    # model_path = os.path.join(path, 'model-150000.h5')
    # fastspeech2 = TFAutoModel.from_pretrained(
    #     config=config, 
    #     pretrained_path=model_path, 
    #     is_build=True, # don't build model if you want to save it to pb. (TF related bug)
    #     name="fastspeech2"
    # )
    fastspeech2 = TFFastSpeech2(config=config, name="fastspeech2v1",
                            enable_tflite_convertible=True)

    # # Newly added :
    # fastspeech2.setup_window(win_front=6, win_back=6)
    # fastspeech2.setup_maximum_iterations(3000)

    fastspeech2._build()
    weights = os.path.join(path, 'model-150000.h5')
    fastspeech2.load_weights(weights)
    print(fastspeech2.summary())
    return fastspeech2

def load_tacotron(path=OUT_TACOTRON_TFLITE_DIR):
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


def generate_tflite(model, out_file, out_dir):
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


def generate_and_save_tflite(out_file=OUT_TACOTRON_TFLITE_FILE, out_dir=OUT_TACOTRON_TFLITE_DIR):
    fastspeech2 = load_fastspeech2()
    generate_tflite(fastspeech2, OUT_FASTSPEECH2_TFLITE_FILE, OUT_FASTSPEECH2_TFLITE_DIR)
    tacotron = load_tacotron()
    generate_tflite(tacotron, out_file, out_dir)

# Prepare input data.
def prepare_input_tacotron2(input_ids):
    return (tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
            tf.convert_to_tensor([len(input_ids)], tf.int32),
            tf.convert_to_tensor([0], dtype=tf.int32))

def prepare_input_fastspeech2(input_ids):
  input_ids = tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0)
  return (input_ids,
          tf.convert_to_tensor([0], tf.int32),
          tf.convert_to_tensor([1.0], dtype=tf.float32),
          tf.convert_to_tensor([1.0], dtype=tf.float32),
          tf.convert_to_tensor([1.0], dtype=tf.float32))

def infer_fastspeech2(input_text, interpreter, input_details, output_details):
  for x in input_details:
    print(x)
  for x in output_details:
    print(x)
  processor = LJSpeechProcessor(None, symbols=LJSPEECH_SYMBOLS)
  input_ids = processor.text_to_sequence(input_text.lower())
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
  input_data = prepare_input_fastspeech2(input_ids)
  for i, detail in enumerate(input_details):
    input_shape = detail['shape']
    interpreter.set_tensor(detail['index'], input_data[i])

  interpreter.invoke()

  # The function `get_tensor()` returns a copy of the tensor data.
  # Use `tensor()` in order to get a pointer to the tensor.
  return (interpreter.get_tensor(output_details[0]['index']),
          interpreter.get_tensor(output_details[1]['index']))

def infer_tacotron2(input_text, interpreter, input_details, output_details):
    # processor = LJSpeechProcessor(None, "english_cleaners")
    processor = AutoProcessor.from_pretrained(LJSPEECH_PROCESSOR_JSON)
    input_ids = processor.text_to_sequence(input_text.lower())
    # eos.
    input_ids = np.concatenate([input_ids, [len(LJSPEECH_SYMBOLS) - 1]], -1)
    interpreter.resize_tensor_input(input_details[0]['index'],
                                    [1, len(input_ids)])
    interpreter.allocate_tensors()
    input_data = prepare_input_tacotron2(input_ids)
    for i, detail in enumerate(input_details):
        print(detail)
        input_shape = detail['shape']
        interpreter.set_tensor(detail['index'], input_data[i])

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    return (interpreter.get_tensor(output_details[0]['index']),
            interpreter.get_tensor(output_details[1]['index']))

# COLAB FOR INFERENCE FROM TFLITE MODELS:
# https://colab.research.google.com/drive/1HudLLpT9CQdh2k04c06bHUwLubhGTWxA?usp=sharing

def run_example():
    # Input text of 777 words (4k chars including spaces) used too much memory locally, 
    # chunk to ~350 chars (75 words) max for best quality 
    # 777 words = 12 mb file
    # input_text = "Recent research at Harvard has shown meditating\
    # for as little as 8 weeks, can actually increase the grey matter in the \
    # parts of the brain responsible for emotional regulation, and learning."

    input_text = "Hi everyone, thank you all for being here with us this evening.\
         I just wanted to say a few words about my grandfather. Most of you here knew him as Ata-Khan,\
              but to a few of us in this room, he will always be Babai. When we were kids, \
                  every so often Babai would come to pick us up from school."

    # Load the TFLite model and allocate tensors.
    # path = os.path.join(OUT_TACOTRON_TFLITE_DIR, OUT_TACOTRON_TFLITE_FILE)
    path = os.path.join(OUT_FASTSPEECH2_TFLITE_DIR, OUT_FASTSPEECH2_TFLITE_FILE)
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    melgan = load_mb_melgan() # load_melgan()

    print("Inference start")
    # decoder_output_tflite, mel_output_tflite = infer_tacotron2(
    #     input_text, interpreter, input_details, output_details)
    decoder_output_tflite, mel_output_tflite = infer_fastspeech2(
        input_text, interpreter, input_details, output_details)
    print("Inference end, vocoder start") # fastspeech inference ~150 words per minute
    audio_before_tflite = melgan(decoder_output_tflite)[0, :, 0]
    print("vocoder1 end, vocoder2 start")
    audio_after_tflite = melgan(mel_output_tflite)[0, :, 0]
    print("Audio shape is {}".format(audio_after_tflite.shape))

    # config = os.path.join('./model_files/melgan', 'config.yml')
    config = os.path.join('./model_files/multiband_melgan', 'config.yml')

    with open(config) as f:
        melgan_config = yaml.load(f, Loader=yaml.Loader)
    sampling_rate = melgan_config["sampling_rate"]

    sf.write('audio_before_tflite.wav', audio_before_tflite,  sampling_rate)
    sf.write('audio_after_tflite.wav', audio_after_tflite, sampling_rate)

############ FASTSPEECH2 ###############################

    # fastspeech2 = load_fastspeech2()
    # processor = AutoProcessor.from_pretrained(LJSPEECH_PROCESSOR_JSON)
    # melgan = load_melgan()

    # config = os.path.join('./model_files/melgan', 'config.yml')
    # with open(config) as f:
    #     melgan_config = yaml.load(f, Loader=yaml.Loader)
    # sampling_rate = melgan_config["sampling_rate"]
    # input_ids = processor.text_to_sequence(input_text)

    # mel_before, mel_after, duration_outputs, _, _ = fastspeech2.inference(
    #     input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
    #     speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
    #     speed_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
    #     f0_ratios =tf.convert_to_tensor([1.0], dtype=tf.float32),
    #     energy_ratios =tf.convert_to_tensor([1.0], dtype=tf.float32)
    # )

    # print("mel_before shape = {}, mel_after shape = {}".format(mel_before.shape, mel_after.shape))
    # audio_before = melgan(mel_before)[0, :, 0]
    # audio_after = melgan(mel_after)[0, :, 0]
    

    # sf.write('audio_before.wav', audio_before,  sampling_rate)
    # sf.write('audio_after.wav', audio_after, sampling_rate)
