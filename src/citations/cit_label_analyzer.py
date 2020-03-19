cit_labelled_path = "../../output/citations/topics_cits_labelled.txt"

cit_topic_list = []
with open(cit_labelled_path, "r", encoding="utf-8") as input_file:
    lines = input_file.readlines()
    for line in lines:
        cit_topic_list.append(line.split(" -> ")[1][1:-2].split(", "))

new_cit_topic_list = []
for cit in cit_topic_list:
    tmp_list = []
    for c in cit:
        if c != "None":
            tmp_list.append(int(c[1:-1]))
        else:
            tmp_list.append(None)
    new_cit_topic_list.append(tmp_list)

cit_topic_list = new_cit_topic_list

ditributions = []
for cit in cit_topic_list:
    tmp = [1 for i in range(40)]
    for c in cit:
        if c is None:
            for i in range(len(tmp)):
                tmp[i] +=1
        else:
            tmp[c] += 1
    ditributions.append(tmp)

normalized = []
for d in ditributions:
    normalized.append([float(i) / sum(d) for i in d])

for n in normalized:
    print(n)
