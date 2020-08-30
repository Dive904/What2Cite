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

print(connections)
