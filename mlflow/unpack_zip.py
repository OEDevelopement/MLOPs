import zipfile
import os

# unpack ZIP-file
with zipfile.ZipFile("mlflow/data/raw/adult-income-dataset.zip", "r") as zip_ref:
    zip_ref.extractall("data/raw")

# delete ZIP-file
os.remove('mlflow/data/raw/adult-income-dataset.zip')