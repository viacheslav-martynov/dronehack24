"""
Functions to save and write data from local storage
"""
import os
import json
import pandas as pd
import shutil
from datetime import datetime
from pathlib import Path
import zipfile

STORAGE_PATH = 'storage'

def init_storage():
    os.makedirs(STORAGE_PATH, exist_ok=True)
    os.makedirs(Path(STORAGE_PATH, 'videos'), exist_ok=True)
    os.makedirs(Path(STORAGE_PATH, 'timelines'), exist_ok=True)
    meta = Path(STORAGE_PATH, 'meta.csv')
    if not meta.is_file():
        meta = pd.DataFrame(
            columns=[
                'experiment',
                'datetime',
                'filename',
                'model',
                'conf',
            ])
        meta.to_csv(Path(STORAGE_PATH, 'meta.csv'), sep=';', index=False)

def save_directory_as_archive(dirpath, zip_file_path):
    dirpath = Path(dirpath)
    Path.unlink(Path(zip_file_path), missing_ok=True)
    with zipfile.ZipFile(zip_file_path, 'a') as zip_ref:
        for filename in os.listdir(dirpath):
            zip_ref.write(Path(dirpath,filename))

def save_video_bytes(video_bytes, video_filename):
    with open(Path(video_filename), 'wb') as f:
        f.write(video_bytes)

def read_video_bytes(video_filename):
    with open(Path(video_filename), 'rb') as f:
        bytes = f.read()
    return bytes

def get_video_bytes_from_storage(video_filename):
    with open(Path(STORAGE_PATH, 'videos', video_filename), 'rb') as f:
        bytes = f.read()
    return bytes

def append_row(df, row):
    return pd.concat([
                df, 
                pd.DataFrame([row], columns=row.index)]
           ).reset_index(drop=True)

def save_file_to_storage(source_filname, dest_filename, model_name, conf, timestamp):
    dest_filename = Path(dest_filename)
    # copy file to storage
    i = 0
    while Path(STORAGE_PATH, 'videos', f"{dest_filename.stem}_pred_{i}{dest_filename.suffix}").is_file():
        i += 1
    dest_filename = Path(f"{dest_filename.stem}_pred_{i}{dest_filename.suffix}")
    shutil.copy(
        source_filname,
        Path(
            STORAGE_PATH,
            'videos',
            dest_filename,
        )
    )

    # add file information to metadata
    meta_path = Path(STORAGE_PATH, 'meta.csv')

    metadata = pd.read_csv(meta_path, sep=';')
    exp_id = len(metadata)
    # now = datetime.now()
    metadata = append_row(
        metadata, 
        pd.Series(
            {
                'experiment':exp_id,
                'datetime': timestamp.strftime("%d/%m/%Y %H:%M:%S"),
                'filename': dest_filename,
                'model': model_name,
                'conf': conf,
            }
        )
    )
    metadata.to_csv(Path(STORAGE_PATH, 'meta.csv'), sep=';', index=False)

    return dest_filename

def reset_storage():
    shutil.rmtree(STORAGE_PATH, ignore_errors=True)
    init_storage()

def read_storage_metadata():
    meta_path = Path(STORAGE_PATH, 'meta.csv')
    metadata = pd.read_csv(meta_path, sep=';')
    # check integrity
    filenames_from_dir = os.listdir(Path(STORAGE_PATH, 'videos'))
    filenames_from_table = list(metadata['filename'])
    # print(filenames_from_dir)
    # print(filenames_from_table)
    if sorted(filenames_from_dir) != sorted(filenames_from_table):
        reset_storage()
    experiments = metadata['experiment']
    datetimes = metadata['datetime']
    filenames = metadata['filename']
    model_names = metadata['model']
    confs = metadata['conf']
    return experiments, datetimes, filenames, model_names, confs

def save_timeline_to_storage(timeline_items, video_filename):
    timeline_filename = Path(video_filename).with_suffix(".json")
    timline_path = Path(STORAGE_PATH, 'timelines', timeline_filename)
    with open(timline_path, 'w') as f:
        json.dump(timeline_items, f)

def read_video_timeline(video_filename):
    timeline_filename = Path(video_filename).with_suffix(".json")
    timline_path = Path(STORAGE_PATH, 'timelines', timeline_filename)
    with open(timline_path, 'r') as openfile:
        json_object = json.load(openfile)
    return json_object

def delete_record_by_id(record_id):
    meta_path = Path(STORAGE_PATH, 'meta.csv')
    metadata = pd.read_csv(meta_path, sep=';')
    exp_data = metadata.iloc[record_id]

    metadata = metadata.drop([record_id], axis=0)
    Path.unlink(Path(STORAGE_PATH, 'videos', exp_data['filename']), missing_ok=True)
    Path.unlink(Path(STORAGE_PATH, 'timelines', exp_data['filename']).with_suffix('.json'), missing_ok=True)
    metadata['experiment'] = list(range(len(metadata)))
    metadata.to_csv(Path(STORAGE_PATH, 'meta.csv'), sep=';', index=False)
