import json


f = open("final_sym.txt", "r")
final = map(str.strip, f.readlines())

with open('data.json', 'w') as outfile:
    json.dump(final, outfile)