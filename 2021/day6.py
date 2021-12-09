from os import read
from utils import read_input
import pandas as pd
import math

file = read_input("day6")

testdata = """3,4,3,1,2"""
testdata = [x for x in testdata.splitlines()]

def part_one(days_needed, test=False):
    if test:
        data = testdata
    else:
        data = file
    data = [int(x) for x in data[0].split(',')]

    day = 0
    lanternfish = data
    # print(f"Initial state: {lanternfish}")
    while True:
        if day == days_needed:
            break
        
        new_lanternfish = []
        for fish in lanternfish:
            fish -= 1

            if fish < 0:
                fish = 6
                new_lanternfish.append(8)  # a new fish

            new_lanternfish.append(fish)
        lanternfish = list(new_lanternfish)
        day += 1
        # print(f"After {day} days: {lanternfish}")

    print(f"Fish after {days_needed} days: {len(lanternfish)}")

def part_two(days_needed, test=False):
    if test:
        data = testdata
    else:
        data = file
    data = [int(x) for x in data[0].split(',')]
    
    lanternfish = [0] * 9

    for no in data:
        lanternfish[no] += 1

    for _ in range(days_needed):
        day_zero = lanternfish.pop(0)
        lanternfish[6] += day_zero
        lanternfish.append(day_zero)

    print(f'Fish after {days_needed} days: {sum(lanternfish)}')

if __name__ == "__main__":
    part_one(days_needed=80)
    part_two(days_needed=256)