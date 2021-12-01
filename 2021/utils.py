def read_input(filename):
    """Opens file and returns list, seperated by line"""
    file = f'2021/input/{filename}.txt'
    with open(file, 'r') as f:
        data = f.read().splitlines()
    return data
