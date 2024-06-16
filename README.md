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
streamlit run src/app/app.py --server.enableXsrfProtection false
```