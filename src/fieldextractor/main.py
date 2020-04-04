# This script is used to extract all paper from initial dataset for a specific Field Of Study

import os
import gzip
import json
import shutil

from src.fieldextractor import utils
from src.fileutils import file_abstract

fos = "Computer Science"
dir_path = "D:\\Università\\Corsi\\Tesi\\SemanticScolarDataset"
year = 2010

files = os.listdir(dir_path)[2:-1]
for inputfile in files:
    print("INFO: Switching to archive " + inputfile)

    file_path = "D:\\Università\\Corsi\\Tesi\\SemanticScolarDataset\\" + inputfile
    filecontent_name = "C:\\Users\\Davide\\Desktop\\semanticdatasetextracted\\" + inputfile[:-3] + "_extracted"

    print("INFO: Extracting archive", end="... ")
    with gzip.open(file_path, 'rb') as f_in:
        with open(filecontent_name, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print("Done ✓")

    print("INFO: Loading JSON and filtering", end="... ")
    result_dic = []
    for line in open(filecontent_name, 'r', encoding="utf-8"):
        d = json.loads(line)
        if utils.apply_filter(d, fos, year):
            d["paperAbstract"] = d["paperAbstract"].replace("\n", "")
            d["paperAbstract"] = d["paperAbstract"].replace("    ", "")
            d["paperAbstract"] = d["paperAbstract"].replace("   ", "")
            d["paperAbstract"] = d["paperAbstract"].replace("  ", "")
            if utils.check_if_abstract_is_english(d["paperAbstract"]):
                result_dic.append(d)

    print("Done ✓")

    print("INFO: Writing output file", end="... ")
    file_abstract.txt_abstract_creator(filecontent_name, result_dic)
    print("Done ✓", end="\n\n")

    os.remove(filecontent_name)
