""" 
A one-off script to find all MP3 files within the
directory and concatenate them together.

Useful when recovering from errors during the normal
convert.py pipeline, without actually running the entire
pipeline again.

$ python3 mp3_concatenator.py

TODO(Brian-Lam): Replace all concat code with a OS System
call to use ffmpeg
e.g.
ffmpeg -i "concat:file1.mp3|file2.mp3" -acodec copy output.mp3
"""

from pydub import AudioSegment
import os

def find_all_mp3_files():
    for root, dirs, files in os.walk('.'):
        for filename in files:
            filepath = os.path.join(root, filename)
            # Don't check subdirectories.
            if filepath.count("/") > 1:
                continue
            if os.path.splitext(filename)[1] == ".mp3":
                yield filepath

combined = AudioSegment.empty()

for chunk_mp3_filepath in find_all_mp3_files():
    print("Concatenating: " + chunk_mp3_filepath)
    chunk_sound_segment = AudioSegment.from_mp3(chunk_mp3_filepath)
    combined += chunk_sound_segment

combined.export("concatenated.mp3", format="mp3")