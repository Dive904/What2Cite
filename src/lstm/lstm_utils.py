import re


def preprocess_text(s):
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', s)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence
