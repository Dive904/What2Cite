from spellchecker import SpellChecker


def apply_filter(d):
    return "paperAbstract" in d.keys() and "title" in d.keys() and "fieldsOfStudy" in d.keys() and "sources" in d.keys()


def check_if_abstract_is_english(abstract):