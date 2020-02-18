# This script is used to count extracted text number

import os
import io

dir_path = "C:\\Users\\Davide\\Desktop\\TesiAPP\\"
files = os.listdir(dir_path)
total_count = 0

for file in files:
    print("INFO: Counting text from " + file, end="... \n")
    tmp_count = 0
    with io.open(dir_path + file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("---"):
                tmp_count += 1
    total_count += tmp_count
    print("INFO: Text in file: " + str(tmp_count) + ". Total count: " + str(total_count), end="\n\n")
