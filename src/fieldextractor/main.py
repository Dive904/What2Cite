# This script is used to extract all paper for a specific Field Of Study

import os
import gzip
import json
import shutil

from src.fieldextractor import utils
from src.fileutils import file_abstract_creator

fos = "Computer Science"
dir_path = "D:\\Università\\Corsi\\Tesi\\SemanticScolarDataset"


files = os.listdir(dir_path)[2:-1]
inputfile = files[0]
print("INFO: Switching to archive " + inputfile)

file_path = "D:\\Università\\Corsi\\Tesi\\SemanticScolarDataset\\" + inputfile
filecontent_name = "../../output/" + inputfile[:-3] + "_extracted"

print("INFO: Extracting archive", end="... ")
with gzip.open(file_path, 'rb') as f_in:
    with open(filecontent_name, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
print("Done ✓")

print("INFO: Loading JSON and filtering", end="... ")
result_dic = []
# data = [json.loads(line) for line in open(filecontent_name, 'r', encoding="utf-8")]
for line in open(filecontent_name, 'r', encoding="utf-8"):
    d = json.loads(line)
    if utils.apply_filter(d):
        result_dic.append(d)

print("Done ✓")

print("INFO: Writing output file", end="... ")
file_abstract_creator.txt_abstract_creator(filecontent_name, result_dic)
print("Done ✓")

os.remove(filecontent_name)
