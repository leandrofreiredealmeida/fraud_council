import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

list_of_files = [
	"data/creditcard.csv",
	"models/.gitkeep",
	"notebooks/01_eda.ipynb",
	"notebooks/02_training.ipynb",
	"app/streamlit_app.py",
	"src/preprocess.py",
	"src/train.py",
	"src/explain.py",
	"README.md",
]

for filepath in list_of_files:
	filepath = Path(filepath)
	filedir, filename = os.path.split(filepath)

	if filedir != "":
		os.makedirs(filedir, exist_ok=True)
		logging.info(f"Creating directory {filedir} for the file: {filename}")
	
	if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
		with open(filepath, "w") as f:
			pass
			logging.info(f"Creating empty file: {filepath}")
	else:
		logging.info(f"{filename} is already exists")