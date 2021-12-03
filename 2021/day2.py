from os import read
from utils import read_input

file = read_input("day2")

testdata = """forward 5
down 5
forward 8
up 3
down 8
forward 2"""
testdata = [x for x in testdata.splitlines()]

def part_one(test=False):
    if test:
        data = testdata
    else:
        data = file

    position = 0
    depth = 0

    for line in data:
        line_split = line.split(' ')
        command = line_split[0].upper()
        movement = int(line_split[1])

        if command == "FORWARD":
            position += movement
        elif command == "UP":
            depth -= movement
        elif command == "DOWN":
            depth += movement
        else:
            raise ValueError(f"Unspecified command found: {command}")

    print(f"Answer for part one: {depth * position}")

def part_two(test=False):
    if test:
        data = testdata
    else:
        data = file

    position = 0
    depth = 0
    aim = 0

    for line in data:
        line_split = line.split(' ')
        command = line_split[0].upper()
        movement = int(line_split[1])

        if command == "FORWARD":
            position += movement
            depth += aim*movement
        elif command == "UP":
            aim -= movement
        elif command == "DOWN":
            aim += movement
        else:
            raise ValueError(f"Unspecified command found: {command}")

    print(f"Answer for part two: {depth * position}")

if __name__ == "__main__":
    part_one(False)
    part_two(False)