from os import read
from utils import read_input

file = read_input("day5")

testdata = """0,9 -> 5,9
8,0 -> 0,8
9,4 -> 3,4
2,2 -> 2,1
7,0 -> 7,4
6,4 -> 2,0
0,9 -> 2,9
3,4 -> 1,4
0,0 -> 8,8
5,5 -> 8,2"""
testdata = [x for x in testdata.splitlines()]

def part_one(test=False):
    if test:
        data = testdata
    else:
        data = file
    
    vents = dict()
    for line in data:
        splitter = line.split(' -> ')
        line_start = splitter[0]
        line_end = splitter[1]

        (x1,y1) = [int(x) for x in line_start.split(',')]
        (x2,y2) = [int(x) for x in line_end.split(',')]

        if x1 == x2 or y1 == y2:
            if (x1,y1) not in vents:
                vents[(x1,y1)] = 1
            else:
                vents[(x1,y1)] += 1

            (x_going, y_going) = (x1,y1)

            while x_going != x2 or y_going != y2:
                if x_going != x2:
                    x_going += 1 if x2 > x_going else -1
                else:
                    y_going += 1 if y2 > y_going else -1

                pos = (x_going,y_going)
                if pos not in vents:
                    vents[pos] = 1
                else:
                    vents[pos] += 1

    counts = [1 for v in vents.values() if v > 1]
    print(f'Answer for part one: {sum(counts)}')

def part_two(test=False):
    if test:
        data = testdata
    else:
        data = file
    
    vents = dict()
    for line in data:
        splitter = line.split(' -> ')
        line_start = splitter[0]
        line_end = splitter[1]

        (x1,y1) = [int(x) for x in line_start.split(',')]
        (x2,y2) = [int(x) for x in line_end.split(',')]

        if (x1,y1) not in vents:
            vents[(x1,y1)] = 1
        else:
            vents[(x1,y1)] += 1
        
        (x_going, y_going) = (x1,y1)

        while x_going != x2 or y_going != y2:
            if x_going != x2:
                x_going += 1 if x2 > x_going else -1
            if y_going != y2:
                y_going += 1 if y2 > y_going else -1

            pos = (x_going,y_going)
            if pos not in vents:
                vents[pos] = 1
            else:
                vents[pos] += 1

    counts = [1 for v in vents.values() if v > 1]
    print(f'Answer for part two: {sum(counts)}')    
        
if __name__ == "__main__":
    part_one()
    part_two()
