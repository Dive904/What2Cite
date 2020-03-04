# This script is used to count extracted text number

import os
import io

path = "../../output/lstmdataset/final.txt"
total_count = 0

with io.open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()
    for line in lines:
        if line.startswith("---"):
            total_count += 1
print("INFO: Total count: " + str(total_count), end="\n\n")
