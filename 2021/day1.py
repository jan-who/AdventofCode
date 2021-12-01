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

def run(test=False):
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

    print(f"Increases: {counter}")

if __name__ == "__main__":
    run()
