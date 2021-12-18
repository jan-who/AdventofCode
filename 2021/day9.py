from utils import read_input
import pandas as pd
import numpy as np

file = read_input("day9")

testdata = """2199943210
3987894921
9856789892
8767896789
9899965678"""
testdata = [x for x in testdata.splitlines()]

def part_one(test=False):
    if test:
        data = testdata
    else:
        data = file
    df = pd.DataFrame([[int(x) for x in y] for y in data])
    
    lowpoints = []
    for row in range(df.shape[0]):
        for col in range(df.shape[1]):
            point = df.iloc[row,col]

            if col > 0 and point >= df.iloc[row,col-1]:
                continue
            if col < df.shape[1]-1 and point >= df.iloc[row,col+1]:
                continue
            if row > 0 and point >= df.iloc[row-1,col]:
                continue
            if row < df.shape[0]-1 and point >= df.iloc[row+1,col]:
                continue
            
            lowpoints.append(point)
    sum_lowpoints = sum([x+1 for x in lowpoints])
    print(f"The sum of low points is {sum_lowpoints}.")

def part_two(test=False):
    if test:
        data = testdata
    else:
        data = file
    df = pd.DataFrame([[int(x) for x in y] for y in data])

    total_points_visited = []
    baisins = []
    for row in range(df.shape[0]):
        for col in range(df.shape[1]):
            coords = (row,col)

            if df.iloc[coords] == 9 or coords in total_points_visited:
                continue

            to_check = []
            if col-1 >= 0 and df.iloc[row,col-1] != 9:
                to_check.append((row,col-1))
            if col+1 < df.shape[1] and df.iloc[row,col+1] != 9:
                to_check.append((row,col+1))
            if row-1 >= 0 and df.iloc[row-1,col] != 9:
                to_check.append((row-1,col))
            if row+1 < df.shape[0] and df.iloc[row+1,col] != 9:
                to_check.append((row+1,col))
            visited = [coords]
            
            while to_check:
                y,x = to_check.pop()

                if (y,x) not in visited:
                    visited.append((y,x))

                if x-1 >= 0 and (y,x-1) not in to_check+visited and df.iloc[(y,x-1)] != 9:
                    to_check.append((y,x-1))
                if x+1 < df.shape[1] and (y,x+1) not in to_check+visited and df.iloc[(y,x+1)] != 9:
                    to_check.append((y,x+1))
                if y-1 >= 0 and (y-1,x) not in to_check+visited and df.iloc[(y-1,x)] != 9:
                    to_check.append((y-1,x))
                if y+1 < df.shape[0] and (y+1,x) not in to_check+visited and df.iloc[(y+1,x)] != 9:
                    to_check.append((y+1,x))

            baisins.append(visited)
            total_points_visited += visited

    # find the length of the largest 3
    baisins_length = [len(x) for x in baisins]
    baisins_length.sort()
    answer = np.prod(baisins_length[-3:])
    print(f'The sum of the largest three is: {answer}')

if __name__ == "__main__":
    part_one()
    part_two()
