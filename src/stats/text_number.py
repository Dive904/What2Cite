# This script is used to count extracted text number

from src.topicmodeller import tm_utils

path = "../../output/lstmdataset/final.txt"
semanticdataset_filname = "C:\\Users\\Davide\\Desktop\\semanticdatasetextracted\\"
total_count = 0

paper_info = tm_utils.extract_paper_info(semanticdataset_filname)

print("Abstract Number: " + str(len(paper_info)))
