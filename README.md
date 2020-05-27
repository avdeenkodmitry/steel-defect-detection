## Необходимые зависимости
* Pytorch 1.1.0
* segmentation-models ()
* python 3.7
* Установить библитеку  [albumentations](https://github.com/albu/albumentations)
* Linux

### Подготовка данных

Необходимо скачать данные, предоставляемые компанией Северсталь.
 [Ссылка](https://www.kaggle.com/c/severstal-steel-defect-detection/data).
Разархивируйте и положите данные в папку `../Input` .  

Структура папки `../Input`:

```
test_images
train_images
sample_submission.csv
train.csv
```

Создайте ссылки на данные:

```bash
cd Kaggle-Steel-Defect-Detection/datasets/steel_data
ln -s ../../../Input/test_images ./
ln -s ../../../Input/train_images ./
ln -s ../../../Input/train.csv ./
ln -s ../../../Input/sample_submission.csv ./
```

### Обучение модели


Обучение классифицирующей модели:

```bash
python train_classify.py --model_name=<model_name> --batch_size=<batch_size> --lr=<lr> --epoch=<epoch>
```

Веса модели будут сохранены вот тут: `checkpoints/<model_name>`

Обучение сегментирующей модели:

```bash
python train_segment.py --model_name=<model_name> --batch_size=<batch_size> --lr=<lr> --epoch=<epoch>
```

Веса модели будут сохранены вот тут: `checkpoints/<model_name>`

В конце необходимо выбрать области на изображении для получения конченого ответа:
```bash
python choose_thre_area.py --model_name=<model_name> --batch_size=<batch_size> 
```

Результаты будут сохранены вот тут:  `checkpoints/<model_name>`


### Создать файл решения в csv формате

```
python create_submission.py
```

**Замечание:** Объединять модели в ансамбль можно с помощью следующего скрипта:

```bash
python utils/cal_thre_area_mean.py
```

