"""
Copies files specified in .txt file fro one directory to another 
"""
import os
from fire import Fire
from pathlib import Path
import shutil

def main(filenames: str, copy_to: str = '.'):
    os.makedirs(copy_to, exist_ok=True)
    copy_to = Path(copy_to)
    with open(filenames, 'r') as f:
        filenames = [Path(line.strip()) for line in f.readlines()]
    for filename in filenames:
        shutil.copyfile(filename, copy_to / filename.name)

if __name__ == '__main__':
    Fire(main)