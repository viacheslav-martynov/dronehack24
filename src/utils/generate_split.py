import os
from fire import Fire
from pathlib import Path
from sklearn.model_selection import train_test_split

def main(imgs_dir: str, output_dir: str = '.'):
    filenames = [imgs_dir+os.sep+filename+"\n" for filename in os.listdir(imgs_dir)]
    # filenames = ['./images'+os.sep+filename+"\n" for filename in os.listdir(imgs_dir)]
    filenames_train, filenames_val = train_test_split(filenames, train_size=0.85, random_state=2024)
    with open(output_dir+os.sep+'train.txt', 'w') as f:
        f.writelines(filenames_train)
    with open(output_dir+os.sep+'val.txt', 'w') as f:
        f.writelines(filenames_val)
    print(filenames_val)

if __name__ == '__main__':
    Fire(main)