import io


def txt_abstract_creator(filename, list_dic):
    with io.open(filename + ".txt", "w", encoding="utf-8") as f:
        for d in list_dic:
            f.write("*** TITLE: " + d["title"] + "\n")
            f.write("*** ABSTRACT: " + d["paperAbstract"] + "\n")
            f.write("---" + "\n")
