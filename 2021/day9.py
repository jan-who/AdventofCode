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


if __name__ == "__main__":
    part_one()
