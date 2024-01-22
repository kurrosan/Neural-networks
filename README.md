Создание нейронной сети для распознавания и анализа эмоций на лицах людей по видеофрагментам
===========
[Данные, нужные для запуска кода](#title1)

[Запуск кода](#title2)  

[Пример видео](#title3)

[Как работает код](#title4)

[Результаты программы](#title5)

## <a id="title1">Данные, нужные для запуска кода</a>
1. Для использования программы нужно скачать архив по [ссылке](https://drive.google.com/drive/folders/1wW3ho4bYHMALsFHjr4cQtoqChso_AkNa)
2. neural_networks.ipynb - основной код

## <a id="title2">Запуск кода</a>  
1. Разархивировать его и использовать папку archive:  
  ![image](https://github.com/kurrosan/Neural-networks/assets/120035199/1d526bc3-1af3-4f2e-9cc4-49c497974276)

   И загрузить паку archive в на Google Диск:
  ![image](https://github.com/kurrosan/Neural-networks/assets/120035199/8b4800a3-9b04-47b5-badb-838e4a779a6e)

   #***!!!В дальнейшем использовать ссылку на папку archive!!!***
   
2.  Подключаем Google Диск:
   ```python
   from google.colab import drive
    drive.mount('/content/gdrive')
   ```
3. Копируем путь этой папки:  
![image](https://github.com/kurrosan/Neural-networks/assets/120035199/2318ff28-6879-4309-9987-43d0ab105091)
4. Добавляем ссылку на папку в этот фрагмент кода:
```python
   train_generator = train_datagen.flow_from_directory(
    'ваша_ссылка',  # Путь к папке с обучающими изображениями.
    target_size=img_size,             
    color_mode='grayscale',           
    batch_size=32,                    
    class_mode='categorical'          
)
```
5. Загрузите видео в Google Colab:  
   ![image](https://github.com/kurrosan/Neural-networks/assets/120035199/66261650-57ad-437a-b703-639b6a35b203)

6. Добавьте ссылку на видео в этот кусок кода:
```python
  video_path = 'ваша_ссылка_на_видео'
  cap = cv2.VideoCapture(video_path)
```
## <a id="title3">Пример видео</a>  
Я загрузила видео [Злой.mp4](https://github.com/kurrosan/Neural-networks/blob/main/%D0%97%D0%BB%D0%BE%D0%B9.mp4) Итоговый результат(1) будет предствлен на основе этого видео.

## <a id="title4">Как работает код</a>
Импортируем нужные библиотеки
```python
 !pip install transformers
```
```python
from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
```
```python
 !pip install tensorflow keras opencv-python numpy pandas
```
```python
 from google.colab import drive
 drive.mount('/content/gdrive')
```
```python
 img_size = (48, 48)
 num_classes = 7
```
Данный код используется для создания объекта модели нейронной сети с использованием библиотеки глубокого обучения Keras, который предоставляет высокоуровневый интерфейс для построения и обучения нейронных сетей.
```python
 model = models.Sequential()
```

```python
 # Добавление слоя свертки с 64 фильтрами размером (3, 3), функцией активации ReLU
# и указанием входной формы данных (48, 48, 1) для изображений размером 48x48 пикселей в оттенках серого.
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)))

# Добавление еще одного слоя свертки с 64 фильтрами размером (3, 3) и функцией активации ReLU.
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Добавление слоя максимального пулинга с окном размером (2, 2),
# который помогает уменьшить размерность данных и выделить основные признаки.
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

# Добавление слоя dropout с коэффициентом отсева 0.25,
# который случайным образом "выключает" некоторые нейроны в процессе обучения,
# чтобы предотвратить переобучение модели.
model.add(layers.Dropout(0.25))
```
```python
 # Добавление слоя свертки с 128 фильтрами размером (3, 3) и функцией активации ReLU.
model.add(layers.Conv2D(128, (3, 3), activation='relu'))

# Добавление еще одного слоя свертки с 128 фильтрами размером (3, 3) и функцией активации ReLU.
model.add(layers.Conv2D(128, (3, 3), activation='relu'))

# Добавление слоя максимального пулинга с окном размером (2, 2),
# который помогает уменьшить размерность данных и выделить основные признаки.
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

# Добавление слоя dropout с коэффициентом отсева 0.25,
# который случайным образом "выключает" некоторые нейроны в процессе обучения,
# чтобы предотвратить переобучение модели.
model.add(layers.Dropout(0.25))
```

```python
 # Добавление слоя свертки с 256 фильтрами размером (3, 3) и функцией активации ReLU.
model.add(layers.Conv2D(256, (3, 3), activation='relu'))

# Добавление еще одного слоя свертки с 256 фильтрами размером (3, 3) и функцией активации ReLU.
model.add(layers.Conv2D(256, (3, 3), activation='relu'))

# Добавление слоя максимального пулинга с окном размером (2, 2),
# который помогает уменьшить размерность данных и выделить основные признаки.
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

# Добавление слоя dropout с коэффициентом отсева 0.25,
# который случайным образом "выключает" некоторые нейроны в процессе обучения,
# чтобы предотвратить переобучение модели.
model.add(layers.Dropout(0.25))
```
```python
 # Добавление слоя Flatten, который преобразует трехмерные данные в одномерный вектор,
# чтобы подготовить данные для полносвязного слоя.
model.add(layers.Flatten())

# Добавление полносвязного слоя с 1024 нейронами и функцией активации ReLU.
model.add(layers.Dense(1024, activation='relu'))

# Добавление слоя dropout с коэффициентом отсева 0.5,
# который случайным образом "выключает" некоторые нейроны в процессе обучения,
# чтобы предотвратить переобучение модели.
model.add(layers.Dropout(0.5))

# Добавление полносвязного слоя с количеством нейронов, равным числу классов (num_classes),
# и функцией активации softmax, которая используется для многоклассовой классификации.
model.add(layers.Dense(num_classes, activation='softmax'))
```
```python
 model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```
```python
 model.summary()
```
![image](https://github.com/kurrosan/Neural-networks/assets/120035199/e143c295-fb86-404e-ad72-7d0e052586ba)

```python
 # Создание объекта ImageDataGenerator для аугментации обучающих данных.
train_datagen = ImageDataGenerator(
    rescale=1./255,             # Нормализация значений пикселей к диапазону [0, 1].
    rotation_range=15,          # Вращение изображений на случайный угол до 15 градусов.
    width_shift_range=0.1,      # Случайный сдвиг по ширине в пределах 10% от ширины изображения.
    height_shift_range=0.1,     # Случайный сдвиг по высоте в пределах 10% от высоты изображения.
    shear_range=0.2,            # Наклон изображений на случайный угол.
    zoom_range=0.2,             # Случайное масштабирование изображений.
    horizontal_flip=True,       # Случайное отражение по горизонтали.
    fill_mode='nearest'         # Способ заполнения пикселей при аугментации.
)

# Создание генератора данных из изображений в указанной директории.
train_generator = train_datagen.flow_from_directory(
    'ваша_ссылка',  # Путь к папке с обучающими изображениями.
    target_size=img_size,              # Размер изображений после загрузки.
    color_mode='grayscale',            # Использование изображений в оттенках серого.
    batch_size=32,                     # Размер пакета (batch size) для обучения.
    class_mode='categorical'           # Тип задачи - многоклассовая классификация.
)
```
```python
 # Определение функции, которая будет управлять изменением learning rate в процессе обучения.
def lr_schedule(epoch):
    initial_learning_rate = 0.001  # Начальная скорость обучения.
    decay = 0.9                    # Коэффициент уменьшения скорости обучения.

    # Уменьшение скорости обучения на 10% каждые 10 эпох, начиная с первой эпохи.
    if epoch % 10 == 0 and epoch != 0:
        return initial_learning_rate * decay
    return initial_learning_rate

# Создание объекта LearningRateScheduler с использованием определенной функции lr_schedule.
lr_scheduler = LearningRateScheduler(lr_schedule)
```
```python
 model.fit(
    train_generator,
    epochs=50,  # Увеличьте количество эпох при необходимости
    steps_per_epoch=train_generator.samples // 32,
    callbacks=[lr_scheduler]
)
# Сохраняем обученную модель
model.save('emotion_model.h5')

```
```python
 from tensorflow.keras.models import load_model

 model = load_model('emotion_model.h5')
```
```python
 import cv2
 import numpy as np
```

```python
 pip install --upgrade opencv-python
```
```python
 emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
```
```python
 import cv2
import numpy as np
import time

# Предполагается, что 'cap' где-то определен в вашем коде
# Например, вы можете его инициализировать так: cap = cv2.VideoCapture('your_video.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Изменение размера изображения до размера, используемого для обучения модели
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_resized = cv2.resize(frame_gray, (48, 48))
    frame_normalized = frame_resized / 255.0

    # Добавление дополнительных размерностей (batch_size и channels) для модели
    input_data = np.expand_dims(np.expand_dims(frame_normalized, axis=-1), axis=0)

    # Предсказание эмоции
    predictions = model.predict(input_data)
    emotion_label_index = np.argmax(predictions)
    emotion_label = emotion_labels[emotion_label_index]

    # Отображение результата на кадре
    cv2.putText(frame, f'Эмоция: {emotion_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Отображение кадра с использованием matplotlib
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title(f'Эмоция: {emotion_label}')
    plt.show()

    print(f'Эмоция: {emotion_label}')

    # Задержка между кадрами (в секундах)
    time.sleep(0.1)

# Освобождение объекта захвата
cap.release()

```
## <a id="title5">Результаты программы</a>
Запустите код, в конце выйдет видео поделенное на несколько фотографий, на которых будет написана эмоция из видео:
1. Итоговый результат:
   ![image](https://github.com/kurrosan/Neural-networks/assets/120035199/905149a8-762d-44b9-bdf1-0bf8cfa68de6)
2. Итоговый результат:
   ![image](https://github.com/kurrosan/Neural-networks/assets/120035199/bf0f23fa-2a30-4738-a014-786639337ce7)








