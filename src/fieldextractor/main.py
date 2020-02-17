# This script is used to extract all paper for a specific Field Of Study

import os
import gzip
import json
import shutil

from src.fieldextractor import utils

fos = "Computer Science"
path = "D:\\Università\\Corsi\\Tesi\\SemanticScolarDataset"

"""
files = os.listdir(path)[2:-1]
for file in files:
    print(file)
"""

file_path = "D:\\Università\\Corsi\\Tesi\\SemanticScolarDataset\\sample-S2-records.gz"
filecontent_name = "../../output/sample-S2-records_extracted"

with gzip.open(file_path, 'rb') as f_in:
    with open(filecontent_name, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

data = [json.loads(line) for line in open(filecontent_name, 'r', encoding="utf-8")]

for d in data:
    if utils.apply_filter(d):
        print(d["paperAbstract"])

# os.remove(filecontent_name)
