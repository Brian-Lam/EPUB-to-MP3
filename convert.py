"""
Python EPUB to MP3 Converter

Calls the Google Cloud Platform Speech-to-Text API 
to convert a book (in epub format) into an MP3 audiobook.

Usage:
$ export GOOGLE_APPLICATION_CREDENTIALS=~/EPUB-To-MP3/GCP-credentials.json
$ python3 convert.py --file="book.epub"
"""

import argparse
import ebooklib
import convert_utils
import os
import time 
from ebooklib import epub
from pydub import AudioSegment
from tqdm import tqdm
from tts_utils import TTS


"""
Basic command line flags
"""
parser = argparse.ArgumentParser(
    description='Parse an epub file and convert it into an MP3 audiobook.'
)
parser.add_argument("--file", 
    nargs="?", 
    default="", 
    help="The filepath of the .epub file to convert"
)
parser.add_argument("--offline", 
    action='store_true',
    help="Whether to use the offline model for parsing. If this flag is passed, offline model is used"
)
args = parser.parse_args()

# The filepath of the eBook to convert.
EPUB_FILEPATH = args.file
ENABLE_OFFLINE = args.offline
if not ENABLE_OFFLINE:
    from google.cloud import texttospeech

"""
Development and Advanced Usage Flags 
"""
# If true, do not actually call the Text to Speech Synthesis API. 
DRY_RUN_MODE = False
# If true, will also write the epub's text to a txt file
OUTPUT_EPUB_TEXT_TO_TEST_FILE = False
# This script will break down the eBook into chunks that will be 
# sent to Google Cloud's text-to-speech API. This value will determine
# the approximate maximum number of characters per chunk.
MAX_CHUNK_LENGTH = 3000
# The maximum number of chunks to convert, so I don't wind up
# getting a massive Google Cloud Text-to-Speech bill because I
# accidentally converted an encyclopedia.
MAX_CHUNK_COUNT = 1000
# The millisecond delay between chunks when chunks of the audiobook
# are concatenated together.
CHUNK_MILLISECOND_DELAY = 100

"""
Calls the Google Cloud Speech to Text API and returns the filename
for an MP3 recording for a given text string, specified in `text`.
If running in DRY_RUN_MODE, the API will not be called - returns None
instead.
"""
def GenerateAudioContentForText(text):
    # Instantiates a client
    client = texttospeech.TextToSpeechClient()

    # Set the text input to be synthesized
    synthesis_input = texttospeech.SynthesisInput(text = text)

    # Build the voice request, select the language code ("en-US") and the ssml
    # voice gender ("neutral")
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    # Try at most five times to retrieve the text-to-speech output from Google
    # Cloud Platofrm.
    for attempt in range(5):
        try:
            # Perform the text-to-speech request on the text input with the selected
            # voice parameters and audio file type
            response = client.synthesize_speech(
                input=synthesis_input, voice=voice, audio_config=audio_config
            )

            # Use the current timestamp to create the MP3 filename.
            timestamp_string = str(convert_utils.current_milli_time())
            audio_filename = timestamp_string + ".mp3"

            # The response's audio_content as binary.
            with open(audio_filename, "wb") as out:
                # Write the response to the output file.
                out.write(response.audio_content)
                print('eBook chunk written to file ' + audio_filename)

            return audio_filename
        except Exception as exception:
            print("Received exception from Google Cloud Platform.")
            print(exception)
            print("Will retry five times.")
        else:
            break
    


"""
Parses an epub file and returns the text contained within it,
stripping XML and HTML tags. 
"""
def epub_to_text():
    book = epub.read_epub(EPUB_FILEPATH)

    epub_content_list = []
    for text in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        html_content = text.get_content().decode("utf-8")
        text_content = convert_utils.chapter_to_text(html_content)
        epub_content_list += text_content

    epub_content_string = "".join(epub_content_list)

    # For debugging only: Write the text contents to file
    if OUTPUT_EPUB_TEXT_TO_TEST_FILE:
        f = open(EPUB_FILEPATH + "_test.txt", "a")
        f.write(epub_content_string)
        f.close()

    return epub_content_string

"""
Breaks down a book into chunks of ~MAX_CHUNK_LENGTH characters each, not
including spaces.
"""
def full_text_to_chunks(full_text):
    chunks = []

    words = full_text.split(" ")

    current_chunk_array = []
    current_chunk_length = 0

    # Metrics for approximating GCP billing. 
    total_word_count = 0
    total_character_count = 0

    for word in words:
        total_word_count += 1 
        # Google Cloud Platform counts spaces when considering characters.
        word_character_count = len(word) + 1
        total_character_count += word_character_count

        # Add word to current chunk
        current_chunk_array.append(word)
        current_chunk_length += word_character_count

        # If this current chunk is too large, finalize it.
        if (current_chunk_length > MAX_CHUNK_LENGTH):
            chunks.append(" ".join(current_chunk_array))
            current_chunk_array = []
            current_chunk_length = 0

    # Finalize the last chunk.
    chunks.append(" ".join(current_chunk_array))

    # Print metrics.
    print("Total words: " + str(total_word_count))
    print("Total characters: " + str(total_character_count))
    print("Book separated into " + str(len(chunks)) + " chunks.")
    
    return chunks

"""
Concatenate the MP3 files, specified as a list of filepath strings
in `chunk_mp3_filepaths`, and saves the combined MP3.
"""
def merge_chunk_mp3s(chunk_mp3_filepaths):
    print("Merging chunks.")
    combined = AudioSegment.empty()

    for chunk_mp3_filepath in chunk_mp3_filepaths:
        chunk_sound_segment = AudioSegment.from_mp3(chunk_mp3_filepath)
        combined += chunk_sound_segment
        # Artificially add a delay between chunks.
        combined += AudioSegment.silent(duration=CHUNK_MILLISECOND_DELAY)

    combined.export(EPUB_FILEPATH.replace(".epub", ".mp3"), format="mp3")

"""
Deletes the MP3 chunk files listed in `chunk_mp3_filepaths`.
"""
def delete_chunk_mp3s(chunk_mp3_filepaths):
    for chunk_mp3_filepath in chunk_mp3_filepaths:
        os.remove(chunk_mp3_filepath)

""" 
Converts eBook text chunks into MP3 files and returns the filepath
for each MP3 file.
"""
def convert_text_chunks_to_speech(ebook_chunks, offline=False):
    chunks_converted = 0
    chunk_mp3_filepaths = []
    for chunk in ebook_chunks:
        if (chunks_converted < MAX_CHUNK_COUNT):
            chunk_mp3_filepath = None
            if not offline:
                chunk_mp3_filepath = GenerateAudioContentForText(chunk)
            else:
                chunk_mp3_filepath = GenerateAudioContentForText(chunk)
                # TODO @allen-n: add offline generation
            chunk_mp3_filepaths.append(chunk_mp3_filepath)
        
        chunks_converted += 1
    
    print("All chunks converted.")
    return chunk_mp3_filepaths

""" 
Run the program.
"""
if __name__ == '__main__':
    tts = TTS()
    arr = ["Hi everyone, thank you all for being here with us this evening.", "I just wanted to say a few words about my grandfather. Most of you here knew him as Ata-Khan,",
           "but to a few of us in this room, he will always be Babai. When we were kids", "every so often Babai would come to pick us up from school."]
    for i, item in enumerate(tqdm(arr)):
        tts.run_inference(item, "audio/Testfile{}".format(i))

    # ebook_text = epub_to_text()
    # ebook_chunks = full_text_to_chunks(ebook_text)

    # if not DRY_RUN_MODE:
    #     chunk_mp3_filepaths = convert_text_chunks_to_speech(ebook_chunks)
    #     merge_chunk_mp3s(chunk_mp3_filepaths)
    #     delete_chunk_mp3s(chunk_mp3_filepaths)
