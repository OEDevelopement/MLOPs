import zipfile
import os

# unpack ZIP-file
with zipfile.ZipFile("../mlflow/data/raw/adult-income-dataset.zip", "r") as zip_ref:
    zip_ref.extractall("../mlflow/data/raw")

# delete ZIP-file after unpacking
os.remove('../mlflow/data/raw/adult-income-dataset.zip')
