import pandas as pd

from src.topicmodeller import tm_utils
from src.citations import cit_utils

input_file = "../../output/official/cit_topic_keywords.csv"
out_filename = "../../output/citations/topics_cits.txt"
number_words = 20

df = pd.read_csv(input_file)
df_list = tm_utils.convert_df_to_list(df, number_words)

final_list = []
for f in df_list:
    count_t = 0
    tmp_list = []
    prec = 0
    total_energy = cit_utils.get_total_energy(f)
    t = total_energy / 2
    for x in f:
        first = x[0]
        second = x[1]
        if prec == second:
            tmp_list.append(first)
        elif count_t < t:
            tmp_list.append(first)
            count_t += second
            prec = second
    final_list.append(tmp_list)

with open(out_filename, "w") as out_file:
    for i in range(len(final_list)):
        out_file.write(str(i) + " ->")
        for x in final_list[i]:
            out_file.write(" " + x)
        out_file.write("\n")

