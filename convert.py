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
parser.add_argument("--local", 
    action='store_true',
    help="Whether to use the local model for parsing. If this flag is passed, local model is used"
)
args = parser.parse_args()

# The filepath of the eBook to convert.
EPUB_FILEPATH = args.file
ENABLE_LOCAL = args.local
if ENABLE_LOCAL:
    from tts_utils import TTS
    import re    
else:
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
    
def eupub_to_chapters():
    """Parse a .epub file and return an array of strings
    corresponding to the chapters of the epub.

    Returns:
        list(str): A list of string objects, each of which corresponds to 
        a chapter of the .epub file, i.e. ["chapter 1 text", "chapter 2 text", ...]
    """    
    path = os.path.abspath(EPUB_FILEPATH)
    print(path)
    book = epub.read_epub(path)

    chapter_texts = []
    for text in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        html_content = text.get_content().decode("utf-8")
        chapter_texts.append(html_content)
    
    full_text = "".join(chapter_texts)
    full_text_chapters = convert_utils.get_all_chapters(full_text) 
    print("{} Chapters found in ebook".format(len(full_text_chapters)))


    # NOTE: For debugging only - Write the text contents to file
    if OUTPUT_EPUB_TEXT_TO_TEST_FILE:
        f = open("chapter_test.txt", "w")
        for i, chapter in enumerate(full_text_chapters):
            # For each chapter, prepend `!!Chapter{Chapter #}{newline}{chapter text}` 
            # for readability
            f.write("!!Chapter {}:\n{}".format(i+1,chapter))
        f.close()

    return full_text_chapters

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

def chapters_to_chunks(chapters, num_sentences=4):
    """Breaks down a book into chunks of ~MAX_CHUNK_LENGTH characters each, not
        including spaces.

    Args:
        chapters (list(str)): array of strings corresponding to chapter text
        num_sentences (int, optional): The number of sentences to add to each chunk. Defaults to 4.

    Returns:
        list(list(str)): a list lists of the form [["string chunks", "of chapter 1], ["string chunks", "of chapter 2"], ...]
    """    
    chunks = []
    pat = re.compile(r'([^\.!?]*[\.!?])', re.M)  # match sentences
    for chapter in chapters:
        sentences = pat.findall(chapter)
        out_sentences = []
        for i in range(0, len(sentences), num_sentences):
            out_sentences.append(
                u"".join(sentences[i: min(len(sentences), i+num_sentences)]))
        chunks.append(out_sentences)

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
def convert_text_chunks_to_speech(ebook_chunks):
    chunks_converted = 0
    chunk_mp3_filepaths = []
    for chunk in ebook_chunks:
        if (chunks_converted < MAX_CHUNK_COUNT):
            chunk_mp3_filepath = GenerateAudioContentForText(chunk)
            chunk_mp3_filepaths.append(chunk_mp3_filepath)
        
        chunks_converted += 1
    
    print("All chunks converted.")
    return chunk_mp3_filepaths

def merge_wavs_to_mp3(out_filename, path=".", ms_delay=100):
    """Merge all the .wav files at location `path` in the filesystem into a single mp3. 
    Add `ms_delay` milliseconds of delay between concatenated files.

    Args:
        out_filename (str): The name of the output file, to be located at {path}/{out_filename}. The filename should end in .mp3
        path (str, optional): The string representing the path to the .wav files to be merged. Defaults to ".".
        ms_delay (int, optional): The milliseconds of delay between merged .wav files in the final mp3. Defaults to 100.
    """    
    combined_mp3 = AudioSegment.empty()
    path = os.path.abspath(path)
    print("Joining audio for {}".format(out_filename))
    for filename in tqdm(sorted(os.listdir(path))):
        if os.path.splitext(filename)[1] == ".wav":
            combined_mp3 += AudioSegment.from_wav(os.path.join(path, filename))
            combined_mp3 += AudioSegment.silent(duration=ms_delay)
           
    out = os.path.abspath(os.path.join(path, out_filename))
    combined_mp3.export(out, format="mp3")

    # Clean up
    for filename in os.listdir(path):
        if os.path.splitext(filename)[1] == ".wav":
            os.remove(os.path.join(path, filename))

def local_text_chunks_to_speech(ebook_chapter_chunks):
    """Generate a mp3 audio file for each chapter of input ebook strings. Files are written to the 'audio' folder.

    Args:
        ebook_chapter_chunks (list(list(str))): 
            a list lists of the form [["string chunks", "of chapter 1], ["string chunks", "of chapter 2"], ...]

    """    
    tts = TTS()
    if 'audio' not in os.listdir('.'):
        os.mkdir('audio')
    path = os.path.abspath('./audio')
    
    for i, chapter in enumerate(ebook_chapter_chunks):

        print("Generating speech for chapter {}".format(i))
        for j, sentence in enumerate(tqdm(chapter)):
            p = os.path.join(path, "chapter_{}_audio_{}".format(i,j))
            tts.run_inference(sentence, p)
        merge_wavs_to_mp3("Chapter{}.mp3".format(i), path, CHUNK_MILLISECOND_DELAY)

    print("All chunks converted and mp3s written to {}.".format(path))

""" 
Run the program.
"""
if __name__ == '__main__':
    if ENABLE_LOCAL:
        ebook_chapter_text = eupub_to_chapters()
        ebook_chunks = chapters_to_chunks(ebook_chapter_text)
        local_text_chunks_to_speech(ebook_chunks)
    else:
        ebook_text = epub_to_text()
        ebook_chunks = full_text_to_chunks(ebook_text)

        if not DRY_RUN_MODE:
            chunk_mp3_filepaths = convert_text_chunks_to_speech(ebook_chunks)
            merge_chunk_mp3s(chunk_mp3_filepaths)
            delete_chunk_mp3s(chunk_mp3_filepaths)
