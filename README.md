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
Этот код создает и обучает модель глубокого обучения для распознавания эмоций на лицах в видеопотоке или видеофайле. Используется архитектура сверточной нейронной сети с последовательными слоями, такими как сверточные слои, слои пулинга и полносвязные слои. Применяются методы активации ReLU и слои Dropout для предотвращения переобучения. Модель компилируется с оптимизатором Adam, функцией потерь categorical crossentropy и метрикой точности. Используется ImageDataGenerator для аугментации обучающих изображений и создания генератора данных. Создается функция lr_schedule, которая управляет изменением скорости обучения в процессе обучения. Модель обучается с использованием генератора данных и определенного расписания скорости обучения. Количество эпох установлено в 50. 


  #***!!!Обратите внимание, что для корректной работы кода требуется выполнение в среде, поддерживающей библиотеки TensorFlow и OpenCV.!!!***


## <a id="title5">Результаты программы</a>
Запустите код, в конце выйдет видео поделенное на несколько фотографий, на которых будет написана эмоция из видео:
1. Итоговый результат:
   ![image](https://github.com/kurrosan/Neural-networks/assets/120035199/905149a8-762d-44b9-bdf1-0bf8cfa68de6)
2. Итоговый результат:  
   ![image](https://github.com/kurrosan/Neural-networks/assets/120035199/bf0f23fa-2a30-4738-a014-786639337ce7)








