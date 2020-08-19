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
for i in range(len(paper_info_2010)):
    paper_info_2010[i]["keep"] = True
    citations = paper_info_2010[i]["outCitations"]
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
        if paper_dic.get(paper_info["id"]) is not None:
            paper_dic[paper_info["id"]] = True
    paper_info_all = None
    gc.collect()
    print("Done ✓", end="\n\n")

print("INFO: Looking for close dataset", end="... ")
to_skip = 0
for i in range(len(paper_info_2010)):
    citations = paper_info_2010[i]["outCitations"]
    count = 0
    for c in citations:
        value = paper_dic.get(c)
        if value:
            count += 1
    citations_len = len(citations)
    if count < ((citations_len / 3) * 2):  # 2/3 of citation in the dataset
        to_skip += 1
        paper_info_2010[i]["keep"] = False

print("Done ✓", end="\n\n")

print("INFO: Number of paper to skip: " + str(to_skip))
print("INFO: Number of remaining papers: " + str(paper_info_2010_len - to_skip), end="\n\n")

print("INFO: Writing output file", end="... ")
with io.open(close_dataset, "w", encoding="utf-8") as f:
    for elem in paper_info_2010:
        if elem["keep"]:
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
