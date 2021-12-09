from os import read
from utils import read_input
import math

file = read_input("day7")

testdata = """16,1,2,0,4,2,7,1,2,14"""
testdata = [x for x in testdata.splitlines()]

def part_one(test=False):
    if test:
        data = testdata
    else:
        data = file
    data = [int(x) for x in data[0].split(',')]

    MAX_POSITION = max(data)
    best_fuel_needed = None
    for target_pos in range(MAX_POSITION):
        fuel_needed = 0
        for crab in data:
            fuel_needed += abs(crab-target_pos)

        if best_fuel_needed is None or best_fuel_needed > fuel_needed:
            best_fuel_needed = fuel_needed
    
    print(f'Answer: {best_fuel_needed}')

def part_two(test=False):
    if test:
        data = testdata
    else:
        data = file
    data = [int(x) for x in data[0].split(',')]

    MAX_POSITION = max(data)
    best_fuel_needed = None
    for target_pos in range(5,6):
        fuel_needed = 0
        for crab in data:
            steps = abs(crab-target_pos)
            fuel_crab = sum([x for x in range(steps)])
            print(f'Crab {crab} needs {fuel_crab} for {steps} steps')
            fuel_needed += math.factorial(abs(crab-target_pos))

        if best_fuel_needed is None or best_fuel_needed > fuel_needed:
            best_fuel_needed = fuel_needed
    
    print(f'Answer: {best_fuel_needed}')

if __name__ == "__main__":
    part_one()
    part_two(True)
