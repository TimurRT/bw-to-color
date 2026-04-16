from clearml import Dataset, StorageManager

DATASET_NAME = "coco_2017_train"
PROJECT_NAME = "Colorization_GAN"
# Прямая ссылка на официальный архив COCO 2017 Train images
COCO_URL = "http://images.cocodataset.org/zips/train2017.zip"

print(f"=== Скачивание архива {COCO_URL} через StorageManager... ===")
# StorageManager сам распакует архив во временную папку и вернет путь
manager = StorageManager()
local_path = manager.get_local_copy(remote_url=COCO_URL)

print(f"=== Архив распакован в: {local_path} ===")

print("=== 1. Создание задачи датасета в ClearML ===")
dataset = Dataset.create(
    dataset_name=DATASET_NAME,
    dataset_project=PROJECT_NAME,
    description="COCO 2017 Training set (118k images) for BW-to-Color GAN"
)

print("=== 2. Добавление файлов... ===")
# Добавляем все файлы из распакованной папки (там будут *.jpg)
dataset.add_files(path=local_path, wildcard="*.jpg")

print("=== 3. Загрузка на сервер ClearML ===")
dataset.upload(verbose=True)

print("=== 4. Финализация ===")
dataset.finalize()
print(f"Готово! ID датасета: {dataset.id}")