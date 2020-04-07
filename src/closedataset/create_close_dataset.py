import gc
import io

from src.topicmodeller import tm_utils

batch_numbers = [(None, 30), (30, 60), (60, 90), (90, 120), (120, 160), (160, None)]
# batch_numbers = [(None, 30)]

close_dataset = "../../output/closedataset/closedataset.txt"

print("INFO: Extracting 2010 dataset", end="... ")
paper_info_2010 = tm_utils.extract_paper_info("C:\\Users\\Davide\\Desktop\\semanticdatasetextracted\\")
print("Done ✓")

paper_info_2010_len = len(paper_info_2010)
print("INFO: Initializing dictionary from " + str(paper_info_2010_len) + " papers", end="... ")
paper_dic = {}
for paper in paper_info_2010:
    citations = paper["outCitations"]
    for c in citations:
        paper_dic[c] = False
print("Done ✓", end="\n\n")

for batch in batch_numbers:
    print("INFO: Extracting dataset with start " + str(batch[0]) + " and with end " + str(batch[1]), end="... ")
    paper_info_all = tm_utils.extract_paper_info("C:\\Users\\Davide\\Desktop\\semanticdatasetextracted_all\\",
                                                 start=batch[0], end=batch[1])
    print("Done ✓")

    print("INFO: Analyzing", end="... ")
    for paper_info in paper_info_all:
        key = paper_dic.get(paper_info["id"])
        if key is not None:
            paper_dic[key] = True
    paper_info_all = None
    gc.collect()
    print("Done ✓", end="\n\n")

print("INFO: Looking for close dataset", end="... ")
to_skip = []
for i in range(len(paper_info_2010)):
    citations = paper_info_2010[i]["outCitations"]
    count = 0
    for c in citations:
        value = paper_dic.get(c)
        if value:
            count += 1
    if len(citations) == 2:
        if count == 0:
            to_skip.append(paper_info_2010[i]["id"])
    elif len(citations) == 1:
        if count == 0:
            to_skip.append(paper_info_2010[i]["id"])
    elif len(citations) == 0:
        to_skip.append(paper_info_2010[i]["id"])
    else:
        if count < 2:
            to_skip.append(paper_info_2010[i]["id"])
print("Done ✓", end="\n\n")
paper_info_2010 = None
gc.collect()

print("INFO: Number of paper to skip: " + str(len(to_skip)))
print("INFO: Number of remaining papers: " + str(paper_info_2010_len - len(to_skip)), end="\n\n")

print("INFO: Extracting 2010 dataset with exception", end="... ")
paper_info_2010 = tm_utils.extract_paper_info("C:\\Users\\Davide\\Desktop\\semanticdatasetextracted\\",
                                              exception=to_skip)
print("Done ✓")

print("INFO: Writing output file", end="... ")
with io.open(close_dataset, "w", encoding="utf-8") as f:
    for elem in paper_info_2010:
        f.write("*** ID: " + elem["id"] + "\n")
        f.write("*** TITLE: " + elem["title"] + "\n")
        f.write("*** YEAR: " + elem["year"] + "\n")
        f.write("*** ABSTRACT: " + elem["paperAbstract"] + "\n")
        f.write("*** OUTCITATIONS: ")
        for i in range(len(elem["outCitations"])):
            f.write(elem["outCitations"][i])
            if i != len(elem["outCitations"]) - 1:
                f.write(", ")
        f.write("\n")
        f.write("---" + "\n")
print("Done ✓")
