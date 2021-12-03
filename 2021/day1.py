from os import read
from utils import read_input

file = read_input("day1")

testdata = """199
200
208
210
200
207
240
269
260
263"""
testdata = [x for x in testdata.splitlines()]

def part_one(test=False):
    if test:
        data = testdata
    else:
        data = file

    # convert data from str to int
    data_int = [int(x) for x in data]

    previous_no = None
    counter = 0
    for n in data_int:
        if previous_no != None and n > previous_no:
            counter += 1
        previous_no = n

    print(f"Part 1: Increases: {counter}")

def part_two(test=False):
    if test:
        data = testdata
    else:
        data = file

    # convert data from str to int
    data_int = [int(x) for x in data]

    idx = 0
    previous_sum = None
    counter = 0
    while True:
        current_sum = sum(data_int[idx:(idx+3)])

        if previous_sum != None and current_sum > previous_sum:
            counter += 1

        previous_sum = current_sum
        idx += 1

        if len(data_int) <= idx:
            break

    print(f"Part 2: Increases: {counter}")

if __name__ == "__main__":
    part_one()
    part_two()
