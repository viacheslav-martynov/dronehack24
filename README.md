 #  Физики и лирики...

...приветствуют Вас, уважаемый Эксперт, в репозитории с нашим решением задачи 23:   
**Нейросеть для мониторинга воздушного пространства вокруг аэропортов**  
в рамках хакатона Лидеры Цифровой Трансформации 2024.

## Давайте же установим и развернем решение

```bash
git clone https://github.com/viacheslav-martynov/dronehack24.git
```

* Проверим, что вместе с кодом скачались веса моделей: в директории models должны быть несколько файлов с разрешением .pt. Файлов нет? Скачайте репозиторий архивом, либо настройте git LFS, как описано [здесь](https://git-lfs.com), после чего повторите git clone.


### Проще всего запустить решение в докере

Чтобы собрать докер:

```bash
cd dronehack24
docker build -t fiziki_i_liriki:v1 . 
```

Запустим докер командой:

```bash
docker run -d -p 8501:8501 fiziki_i_liriki:v1
```

После запуска на хост-машине на порте 8501 будет доступен прекрасный пользовательский интерфейс.


### Кроме того, можно запустить в виртуальной среде

Установим конду, если она не установлена. Процесс описан на [официальной странице](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html).  

Создаем и активируем виртуальную среду:

```bash
conda create -n fiziki_i_liriki python=3.10
conda activate fiziki_i_liriki
```

Установим зависимости:

```bash
cd dronehack24
pip install -r requirements.txt
pip install -e sahi
pip install -e my_classificationlib
```

Эти команды установили необходимые зависимости. Однако есть нюанс с тем, что они устанавливают не совсем подходящую нам версию OpenCV. Нужный нам пакет есть в conda-forge. Поэтому выполняем:

```bash
pip uninstall --yes opencv-python
conda install --yes -c conda-forge opencv
```

Запуск осуществляется командой
```bash
streamlit run src/app/app.py
```

### Пробросим порт на машину клиента

Чтобы пробросить порт на клиета выполняем команду на машине клиента
```bash
ssh -NfL 8501:localhost:8501 usename@server_ip
```

Теперь у вас по адресу localhost:8501 доступен пользовательский интерфейс!

## Работаем с пользовательским интерфейсом

Перед собой видите страницу, содержащую три основные секции:
* Верху секцию загрузки фото и видео
* Внизу секцию просмотра видеозаписей в хранилище
* Слева секцию настроек

Пройдемся по каждой из них. 

#### Секция загрузки данных

В секции загрузки вы можете загрузить для обработки либо видео в формате **.mp4**, либо **.zip** архив с фотографиями. В архиве должны быть только фотографии и не должно быть других файлов. Фотографии должны быть в корневой директории архива.

После загрузки видео или архива, появится кнопка "Обработать ...". Нажмите её, и модель начнет совершать предсказания на ваших данных. После завершения процесса, вам будет предоставлена возможность скачать архив предсказаний, либо просмотреть видео в хранилище. Видео добавляется в хранилище автоматически. Архивы фото в хранилище не добавляются.

#### Секция хранилища

В секции хранилища представлены видеозаписи, по которым совершались предсказания. Доступна загрузка видеозаписи с отрисовкой детекций, просмотр, и удаление. Помимо этого, в таблице, вы можете увидеть данные о времени загрузки видео и об использованной модели.

#### Секция настроек

В секции настроек в левой части страницы есть возможность выбрать модель для детекции, порог уверенности модели, а также очистить хранилище. В соответствии с ТЗ, по-умолчанию установлен порог уверенности 0.25. Важно: при изменении модели или порога уверенности, пропадает возможность загрузить архив с предсказаниями. Успейте его загрузить до изменения настроек.


Желаем вам приятной работы с нашей системой и хороший впечатлений от детекции! 

По всем интрересующим вопрсам (если вдруг что-то не работает) оперативная связь со мной в Телеграме: @viacheslav_martynov, по номеру телефона +79920128470, по почте slavk.martynov2015@mail.ru
