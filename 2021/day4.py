from os import read
from utils import read_input

file = read_input("day4")

testdata = """7,4,9,5,11,17,23,2,0,14,21,24,10,16,13,6,15,25,12,22,18,20,8,19,3,26,1

22 13 17 11  0
 8  2 23  4 24
21  9 14 16  7
 6 10  3 18  5
 1 12 20 15 19

 3 15  0  2 22
 9 18 13 17  5
19  8  7 25 23
20 11 10 24  4
14 21 16 12  6

14 21 17 24  4
10 16 15  9 19
18  8 23 26 20
22 11 13  6  5
 2  0 12  3  7"""
testdata = [x for x in testdata.splitlines()]

def part_one(test=False):
    if test:
        data = testdata
    else:
        data = file

    random_numbers = data[0]
    
    boards = []
    board = []
    for line in data[2:]:
        print("line: " + line)
        if line == "":
            boards.append(board)
            board = []
        else:
            l = []
            for n in line.split(' '):
                if n != "":
                    l.append(int(n))
            board.append(l)
    print("\n\n")
    print(boards)

    # make 5x5 grid for each board, filled with ones
    # board in 5x5 grid as well
    # for each row/column a list - if drawn_number is in list
    # mark grid in position of row/column with 0
    # sum of unmarked - multiply both grids

if __name__ == "__main__":
    part_one(True)