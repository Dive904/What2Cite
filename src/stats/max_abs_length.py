from src.fileutils import file_abstract

dataset = file_abstract.txt_lstm_dataset_reader("../../output/lstmdataset/final.txt")
max_length = 0
for data in dataset:
    length = len(data["paperAbstract"])
    if length > max_length:
        max_length = length

print("Max Length: " + str(max_length))
