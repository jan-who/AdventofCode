from utils import read_input
import pandas as pd

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

    point_in_baisin = []
    baisins = []
    for row in range(df.shape[0]):
        for col in range(df.shape[1]):
            coords = (row,col)

            if df.iloc[coords] == 9 or coords in point_in_baisin:
                continue

            baisin = []
            to_check = []
            if col-1 >= 0 and df.iloc[col-1,row] != 9:
                to_check.append((col-1,row))
            if col+1 < df.shape[1]+1 and df.iloc[col+1,row] != 9:
                to_check.append((col+1,row))
            if row-1 >= 0 and df.iloc[col,row-1] != 9:
                to_check.append((col,row-1))
            if row+1 < df.shape[0]+1 and df.iloc[col,row+1] != 9:
                to_check.append((col,row+1))
            visited = [coords]

            print(f"point: {coords}; check: {to_check}; visit: {visited}")
            break
            while to_check:
                print(f"to_check: {to_check}")
                print(f"visited: {visited}")
                x,y = to_check.pop()
                if df.iloc[x,y] == 9:
                    continue

                if x <= df.shape[1]-1 and x >= 0 and \
                    y <= df.shape[0]-1 and y >= 0 and \
                        (x,y) not in visited and df.iloc[x,y] != 9:
                    baisin.append((x,y))
                
                if (x,y) not in visited:
                    visited.append((x,y))
                if x-1 >= 0 and (x-1,y) not in to_check:
                    to_check.append((x+1,y))
                if x+1 < df.shape[1]+1 and (x,y-1) not in to_check:
                    to_check.append((x+1,y))
                if y-1 >= 0 and (x-1,y) not in to_check:
                    to_check.append((x,y-1))
                if y+1 < df.shape[0]+1 and (x,y+1) not in to_check:
                    to_check.append((x,y+1))

                point_in_baisin.append(coords)
            baisins.append(baisin)
            print(baisin)
            break

if __name__ == "__main__":
    #part_one()
    part_two(True)
