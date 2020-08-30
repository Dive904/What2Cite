import matplotlib.pyplot as plt

# input
topics_cit_path = "../../output/official/topics_cits.txt"

f = open(topics_cit_path, "r")
lines = f.readlines()

connections = {}
for line in lines:
    l = line[:-1]
    l = l.split()
    id = l[0]
    l = l[2:]

    for elem in l:
        if elem in connections.keys():
            connections[elem].append(id)
        else:
            connections[elem] = [id]

connections_number = {}

for key in connections.keys():
    card = len(connections[key])

    if card in connections_number.keys():
        connections_number[card] += 1
    else:
        connections_number[card] = 1

del connections_number[1]

names = list(connections_number.keys())
values = list(connections_number.values())

names = list(map(lambda x: str(x), names))

fig, axs = plt.subplots()
axs.bar(names, values)
plt.show()
