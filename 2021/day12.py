from os import path
from utils import read_input
import pandas as pd

file = read_input("day12")

testdata = """fs-end
he-DX
fs-he
start-DX
pj-DX
end-zg
zg-sl
zg-pj
pj-he
RW-he
fs-DX
pj-RW
zg-RW
start-pj
he-WI
zg-he
pj-fs
start-RW"""
testdata = [x for x in testdata.splitlines()]

def part_one(test=False):
    if test:
        data = testdata
    else:
        data = file

    cave_map = dict()
    for x in data:
        splitter = x.split('-')
        if splitter[0] not in cave_map:
            cave_map[splitter[0]] = [splitter[1]]
        else:
            cave_map[splitter[0]].append(splitter[1])
    
        if splitter[1] not in cave_map:
            cave_map[splitter[1]] = [splitter[0]]
        else:
            cave_map[splitter[1]].append(splitter[0])

    pathes = []
    next = [['start']]
    
    while next:
        old_path = next.pop()
        last_pos = old_path[-1]
        
        for p in cave_map[last_pos]:
            if p == p.lower() and p in old_path:
                continue

            new_path = old_path + [p]

            if p == "end":
                pathes.append(new_path)
            else:
                next.append(new_path)

    print(f'Found {len(pathes)} pathes')

if __name__ == "__main__":
    part_one()
