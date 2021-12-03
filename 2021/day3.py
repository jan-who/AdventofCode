from os import read
from utils import read_input
import pandas as pd

file = read_input("day3")

testdata = """00100
11110
10110
10111
10101
01111
00111
11100
10000
11001
00010
01010"""
testdata = [x for x in testdata.splitlines()]

def part_one(test=False):
    if test:
        data = testdata
    else:
        data = file
    data = [[int(y) for y in x] for x in data]
    df = pd.DataFrame(data)

    gamma_rate = []
    epsilon_rate = []

    columns = len(df.iloc[0,:])
    for idx in range(columns):
        col_len = len(df.iloc[:,idx])
        col_sum = sum(df.iloc[:,idx])

        if col_sum >= col_len/2:
            gamma_rate.append(1)
            epsilon_rate.append(0)
        else:
            gamma_rate.append(0)
            epsilon_rate.append(1)

    # converting the list to a str, than to a binary number
    gamma_rate_str = ''.join(str(i) for i in gamma_rate)
    gamma_rate_int = int(gamma_rate_str, 2)  # the ,2 lets python know it's binary
    
    epsilon_rate_str = ''.join(str(i) for i in epsilon_rate)
    epsilon_rate_int = int(epsilon_rate_str, 2)

    print(f"Answer for part one: {gamma_rate_int * epsilon_rate_int}")

if __name__ == "__main__":
    part_one(False)
    #part_two(False)