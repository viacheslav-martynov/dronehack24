import os
import torch
import streamlit as st
import shutil
import yaml
import zipfile
from datetime import datetime
from streamlit_timeline import st_timeline
from pathlib import Path
import time

from storage import (
    STORAGE_PATH,
    init_storage, save_video_bytes, read_video_bytes,
    save_file_to_storage, read_storage_metadata,
    save_timeline_to_storage, read_video_timeline,
    reset_storage, get_video_bytes_from_storage,
    delete_record_by_id, save_directory_as_archive)
# from predict import GenericModel
from predict import predict_video, predict_image_directory
from sahi import AutoDetectionModel

# st.set_page_config(layout="wide")
st.set_page_config(page_title = "Физики и лирики", page_icon=":eyes:")

MODELS_PATH = 'models.yaml'

def get_model_info():
    with open(MODELS_PATH) as stream:
        model_data = yaml.safe_load(stream)
    return model_data

# Functions related to storage
def on_more_click(show_more, idx):
    for key in show_more:
        if key == idx:
            show_more[key] = True
        else:
            show_more[key] = False

def on_less_click(show_more, idx):
    show_more[idx] = False

def get_detector_model(detector_model_type, detector_checkpoint, confidence_threshold, device, should_augment):
    #print(detector_checkpoint)
    return AutoDetectionModel.from_pretrained(
                model_type=detector_model_type,
                model_path=detector_checkpoint,
                confidence_threshold=confidence_threshold,
                device=device,
                augment=should_augment,
                agnostic_nms=True)

def get_inference_func(task: str, should_use_sahi: bool):
    """
    task : image, video, archive
    """
    match (task, should_use_sahi):
        case ('video', True):
            return 

def init_session():
    if 'initialised' not in st.session_state:
        init_storage()
        st.session_state['initialised'] = True
        st.session_state['latest_used_model'] = None
        st.session_state['latest_loaded_image'] = None
        st.session_state['latest_loaded_video'] = None
        st.session_state['latest_loaded_archive'] = None
        st.session_state['orig_size'] = True

    if 'timeline' not in st.session_state:
        st.session_state['timeline'] = None
    if 'source_type' not in st.session_state:
        st.session_state['source_type'] = None
    if 'uploaded_file' not in st.session_state:
        st.session_state['uploaded_file'] = None
    if 'should_display_player' not in st.session_state:
        st.session_state['should_display_player'] = False

    if 'model_info' not in st.session_state:
        st.session_state['model_info'] = get_model_info()

    if 'model' not in st.session_state:
        default_model_key = list(st.session_state['model_info'].keys())[0]
        st.session_state['detector_model'] = get_detector_model(
                detector_model_type=st.session_state['model_info'][default_model_key]['model_type'],
                detector_checkpoint=st.session_state['model_info'][default_model_key]['checkpoint'],
                confidence_threshold=0.25,
                device='cuda',
                should_augment=st.session_state['model_info'][default_model_key]['augment']
        )

        if 'classifier_checkpoint' in st.session_state['model_info'][default_model_key]:

            st.session_state['classifier_model'] = torch.load(
                st.session_state['model_info'][default_model_key]['classifier_checkpoint'],
                map_location='cuda'
            )
            st.session_state['classifier_model'].eval()
        else: 
            st.session_state['classifier_model'] = None

        st.session_state['should_use_sahi'] = st.session_state['model_info'][default_model_key]['sahi']
        st.session_state['sahi_res'] = st.session_state['model_info'][default_model_key]['sahi_res']
        # st.session_state['orig_size'] =  st.session_state['model_info'][default_model_key]['mixed_inf']

def display_player(container, start_timestamp, timeline, video_bytes):
    if timeline is None:
        container.video(video_bytes) #displaying the video
    else:
        datetime_at_timeline = datetime.strptime(timeline['start'], "%Y/%m/%d %H:%M:%S")
        start_time = int((datetime_at_timeline - start_timestamp).total_seconds())+4
        container.video(video_bytes, start_time=start_time) #displaying the video

def display_timeline(container, items, **kwargs):
    with container:
        timeline = st_timeline(items, groups=[], options={}, height="300px", **kwargs)
        st.session_state['timeline'] = timeline

@st.cache_data(show_spinner='Подождите, ваше видео обрабатывается...')
def process_video():
    video_bytes = st.session_state['uploaded_file'].getvalue()
    filename = st.session_state['uploaded_file'].name
    st.text(filename)

    save_video_bytes(video_bytes, 'temp.mp4')

    t1 = time.time()
    results_yolo, timeline_items, start_timestamp = predict_video(
        st.session_state['detector_model'],
        'temp.mp4',
        sahi=st.session_state['should_use_sahi'],
        slice_height=st.session_state['sahi_res'],
        slice_width=st.session_state['sahi_res'],
        save_filename='pred.mp4',
        classifier_model=st.session_state['classifier_model'],
        orig_size= st.session_state['orig_size']
        )
    t2= time.time()
    elapsed_time = t2-t1
    processed_video_bytes = read_video_bytes('pred.mp4')
    return processed_video_bytes, timeline_items, start_timestamp, elapsed_time


@st.cache_data(show_spinner='Подождите, ваш архив обрабатывается...')
def process_image_archive():
    shutil.rmtree('temp_archive', ignore_errors=True)
    os.makedirs('temp_archive', exist_ok=True)
    with zipfile.ZipFile(st.session_state['uploaded_file'], "r") as z:
        z.extractall("temp_archive")
    images_dir = Path("temp_archive")
    predict_image_directory(
        detection_model = st.session_state['detector_model'],
        image_dir = images_dir,
        sahi = False,
        should_save_preds = True,
        save_path='archive_labels',
        orig_size= st.session_state['orig_size']
    )
    result_filename = f"{Path(st.session_state['uploaded_file'].name).stem}_results.zip"
    save_directory_as_archive(
        'archive_labels',
        result_filename
    )
    return result_filename

@st.cache_data
def save_processed_video(timeline_items, _model_name, _conf, start_timestamp):
    # save video to storage
    saved_filename = save_file_to_storage('pred.mp4', st.session_state['uploaded_file'].name, _model_name, _conf, start_timestamp)
    save_timeline_to_storage(timeline_items, saved_filename)

def main():
    init_session()

    # Приложение сделано виде одной страницы
    # Лэйаут последовательный. Секции идут одна за другой

    # Сайдбар с настройками
    with st.sidebar:
        model_names = list(st.session_state['model_info'].keys())
        model_name_option = st.selectbox(
            "Какую модель будем использовать для мониторинга?",
            model_names)
        
        confidence_threshold = st.slider("Порог уверенности", min_value=0.1, max_value=1., step=0.05, value=0.25)

        st.session_state['detector_model'] = get_detector_model(
            detector_model_type=st.session_state['model_info'][model_name_option]['model_type'],
            detector_checkpoint=st.session_state['model_info'][model_name_option]['checkpoint'],
            confidence_threshold=confidence_threshold,
            device='cuda',
            should_augment=st.session_state['model_info'][model_name_option]['augment'],
        )

        if 'classifier_checkpoint' in st.session_state['model_info'][model_name_option]:

            st.session_state['classifier_model'] = torch.load(
                st.session_state['model_info'][model_name_option]['classifier_checkpoint'],
                map_location='cuda'
            )
            st.session_state['classifier_model'].eval()
        else: 
            st.session_state['classifier_model'] = None

        st.session_state['should_use_sahi'] = st.session_state['model_info'][model_name_option]['sahi']
        st.session_state['sahi_res'] = st.session_state['model_info'][model_name_option]['sahi_res']
        # st.session_state['orig_size'] =  st.session_state['model_info'][model_name_option]['mixed_inf']

        st.session_state['orig_size'] = st.checkbox('Инференс в исходном разрешении', value=True)

        st.button('Очистить хранилище',
                  help="Очищает хранилище",
                  on_click=reset_storage)

    # Секция загрузки видео

    upload_section = st.container()
    with upload_section:
        st.header('Здесь вы можете загрузить фото/видео для обработки.')
        st.session_state['source_type'] = st.radio(
            "Я загружаю...",
            ["Видео :movie_camera:", "Архив фото :package:"], # "Фото :frame_with_picture:", 
            index=None,
        )
        if st.session_state['source_type'] is not None:
            st.session_state['uploaded_file'] = st.file_uploader(f"Выберите {st.session_state['source_type']}")

        match st.session_state['source_type']:
            case "Фото :frame_with_picture:":
                pass
            case "Видео :movie_camera:":
                if st.session_state['latest_loaded_video'] != st.session_state['uploaded_file']:
                    process_video.clear()
                    save_processed_video.clear()
                st.session_state['latest_loaded_video'] = st.session_state['uploaded_file']
                if st.session_state['uploaded_file'] is not None:
                    if st.button('Обработать видео', key='go_video', on_click=None):
                        process_video.clear()
                        save_processed_video.clear()
                        # To read file as bytes:
                        video_bytes, timeline_items, start_timestamp, elapsed_time = process_video() # TODO return success
                        
                        # if success process
                        save_processed_video(timeline_items, model_name_option, confidence_threshold, start_timestamp)

                        st.text("Видео обработано. Просмотрите его в секции ниже.")
                        st.text(f"Время обработки нейросетью {elapsed_time} сек.")
            case "Архив фото :package:":
                if st.session_state['latest_loaded_archive'] != st.session_state['uploaded_file']:
                    process_image_archive.clear()
                    # save_processed_image_archive.clear()
                st.session_state['latest_loaded_archive'] = st.session_state['uploaded_file']
                if st.session_state['uploaded_file'] is not None:
                    if st.button('Обработать архив', key='go_archive', on_click=None):
                        process_image_archive.clear()
                        result_filename = process_image_archive()
                        with open(result_filename, 'rb') as f:
                            data_bytes = f.read()
                            st.download_button('Скачать архив с предсказаниями', data_bytes, result_filename)
            case _:
                pass
            

    # Историческая секция
    history_section = st.container()
    with history_section:
        st.header('Здесь вы можете посмотреть и загрузить обработанные видео.')

        experiments, datetimes, filenames, model_names, confs = read_storage_metadata()

        if "show_more" not in st.session_state:
            st.session_state["show_more"] = dict.fromkeys(experiments, False)
                    # update show_more
        for exp in experiments:
            if exp not in st.session_state["show_more"].keys():
                st.session_state["show_more"][exp]=False
        show_more = st.session_state["show_more"]

        cols   = st.columns(8)
        fields = ["№", "Время", "Имя файла", "Модель", "Порог", "Просмотр", "Загрузить", "Удалить"]

        # header
        for col, field in zip(cols, fields):
            col.write("**" + field + "**")

        # rows
        for experiment, datetime_record, filename, model_name, conf in zip(experiments, datetimes, filenames, model_names, confs):
            col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
            col1.write(str(experiment))
            col2.write(str(datetime_record))
            col3.write(str(filename))
            col4.write(str(model_name))
            col5.write(str(conf))
            
            placeholder = col6.empty()

            if show_more[experiment]:
                placeholder.button(
                    ":tv:", key=str(experiment) + "_", on_click=on_less_click, args=[show_more, experiment]
                )

                video_timeline = read_video_timeline(filenames[experiment])
                history_player = st.container()
                history_timeline_container = st.container()
                display_timeline(history_timeline_container, video_timeline, key=100)
                display_player(
                    history_player,
                    datetime.strptime(datetime_record, "%d/%m/%Y %H:%M:%S"),
                    st.session_state['timeline'],
                    str(Path(STORAGE_PATH, 'videos', filenames[experiment]))
                    )
            else:
                placeholder.button(
                    ":tv:",
                    key=experiment,
                    on_click=on_more_click,
                    args=[show_more, experiment],
                    type="primary",
                )
            download_placeholder = col7.empty()
            download_placeholder.download_button(':arrow_down:', get_video_bytes_from_storage(filename), filename)

            delete_placeholder = col8.empty()
            delete_placeholder.button(':x:', key=f"b{experiment}", on_click=delete_record_by_id, args=[experiment])

        


if __name__ == "__main__":
    main()
