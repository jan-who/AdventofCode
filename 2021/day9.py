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
    df = pd.DataFrame(data)
    print(df.iloc[0,:])
    for row in range(df.shape[0]):
        for col in range(df.shape[1]):
            print(df.iloc[row,col][0])
    


if __name__ == "__main__":
    part_one(True)
