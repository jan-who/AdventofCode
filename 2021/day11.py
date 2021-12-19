from utils import read_input
import pandas as pd

file = read_input("day11")

testdata = """5483143223
2745854711
5264556173
6141336146
6357385478
4167524645
2176841721
6882881134
4846848554
5283751526"""
testdata = [x for x in testdata.splitlines()]

def part_one(test=False, STEPS=100):
    if test:
        data = testdata
    else:
        data = file
    df = pd.DataFrame([[int(x) for x in y] for y in data])
    
    flashes = 0
    for s in range(STEPS):
        df += 1

        flashing = True if max(df.max(0)) > 9 else False
        flashed = []
        while flashing:
            increased = False
            # go through every element and check whether it flashes or not
            for row in range(df.shape[0]):
                for col in range(df.shape[1]):
                    if df.iloc[row,col] > 9 and (row,col) not in flashed:
                        flashes += 1
                        flashed.append((row,col))
                        # increase energy of adjacent octopuses
                        if row-1 >= 0:
                            df.iloc[row-1,col] += 1
                        if row+1 < df.shape[0]:
                            df.iloc[row+1,col] += 1

                        if col-1 >= 0:
                            df.iloc[row,col-1] += 1
                        if col+1 < df.shape[1]:
                            df.iloc[row,col+1] += 1

                        if row-1 >= 0 and col-1 >= 0:
                            df.iloc[row-1,col-1] += 1
                        if row-1 >= 0 and col+1 < df.shape[1]:
                            df.iloc[row-1,col+1] += 1
                        if row+1 < df.shape[0] and col-1 >= 0:
                            df.iloc[row+1,col-1] += 1
                        if row+1 < df.shape[0] and col+1 < df.shape[1]:
                            df.iloc[row+1,col+1] += 1

                        increased = True

            flashing = True if max(df.max(0)) > 9 else False

            if not increased:
                break

        for row in range(df.shape[0]):
            for col in range(df.shape[1]):
                if df.iloc[row,col] > 9:
                    df.iloc[row,col] = 0

    print(f'Recored {flashes} flashes after {STEPS} steps.')

def part_two(test=False):
    if test:
        data = testdata
    else:
        data = file
    df = pd.DataFrame([[int(x) for x in y] for y in data])
    
    steps = 0
    while True:
        df += 1

        flashing = True if max(df.max(0)) > 9 else False
        flashed = []
        while flashing:
            increased = False
            # go through every element and check whether it flashes or not
            for row in range(df.shape[0]):
                for col in range(df.shape[1]):
                    if df.iloc[row,col] > 9 and (row,col) not in flashed:
                        flashed.append((row,col))
                        # increase energy of adjacent octopuses
                        if row-1 >= 0:
                            df.iloc[row-1,col] += 1
                        if row+1 < df.shape[0]:
                            df.iloc[row+1,col] += 1

                        if col-1 >= 0:
                            df.iloc[row,col-1] += 1
                        if col+1 < df.shape[1]:
                            df.iloc[row,col+1] += 1

                        if row-1 >= 0 and col-1 >= 0:
                            df.iloc[row-1,col-1] += 1
                        if row-1 >= 0 and col+1 < df.shape[1]:
                            df.iloc[row-1,col+1] += 1
                        if row+1 < df.shape[0] and col-1 >= 0:
                            df.iloc[row+1,col-1] += 1
                        if row+1 < df.shape[0] and col+1 < df.shape[1]:
                            df.iloc[row+1,col+1] += 1

                        increased = True

            flashing = True if max(df.max(0)) > 9 else False

            if not increased:
                break

        for row in range(df.shape[0]):
            for col in range(df.shape[1]):
                if df.iloc[row,col] > 9:
                    df.iloc[row,col] = 0

        steps += 1
        if sum(df.sum(0)) == 0:
            break

    print(f'First time all simultaneously flashes after {steps} steps.')

if __name__ == "__main__":
    part_one()
    part_two()
