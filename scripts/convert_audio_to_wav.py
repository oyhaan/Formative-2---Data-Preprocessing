import os
import subprocess

# Directory containing audio files
audio_dir = './data/sound'

# Supported input formats for conversion
input_exts = ['.m4a', '.aac']

def convert_to_wav(input_path, output_path):
    cmd = [
        'ffmpeg',
        '-y',  # Overwrite output files without asking
        '-i', input_path,
        output_path
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Converted: {input_path} -> {output_path}")
    except Exception as e:
        print(f"Failed to convert {input_path}: {e}")

for fname in os.listdir(audio_dir):
    ext = os.path.splitext(fname)[1].lower()
    if ext in input_exts:
        input_path = os.path.join(audio_dir, fname)
        output_path = os.path.join(audio_dir, os.path.splitext(fname)[0] + '.wav')
        convert_to_wav(input_path, output_path)
