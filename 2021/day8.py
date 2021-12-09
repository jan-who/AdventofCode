from utils import read_input

file = read_input("day8")

testdata = """be cfbegad cbdgef fgaecd cgeb fdcge agebfd fecdb fabcd edb | fdgacbe cefdb cefbgd gcbe
edbfga begcd cbg gc gcadebf fbgde acbgfd abcde gfcbed gfec | fcgedb cgb dgebacf gc
fgaebd cg bdaec gdafb agbcfd gdcbef bgcad gfac gcb cdgabef | cg cg fdcagb cbg
fbegcd cbd adcefb dageb afcb bc aefdc ecdab fgdeca fcdbega | efabcd cedba gadfec cb
aecbfdg fbg gf bafeg dbefa fcge gcbea fcaegb dgceab fcbdga | gecf egdcabf bgf bfgea
fgeab ca afcebg bdacfeg cfaedg gcfdb baec bfadeg bafgc acf | gebdcfa ecba ca fadegcb
dbcfg fgd bdegcaf fgec aegbdf ecdfab fbedc dacgb gdcebf gf | cefg dcbef fcge gbcadfe
bdfegc cbegaf gecbf dfcage bdacg ed bedf ced adcbefg gebcd | ed bcgafe cdgba cbgef
egadfb cdbfeg cegd fecab cgb gbdefca cg fgcdab egfdb bfceg | gbdfcae bgc cg cgb
gcafb gcf dcaebfg ecagb gf abcdeg gaef cafbge fdbac fegbdc | fgae cfgab fg bagce"""
testdata = [x for x in testdata.splitlines()]

def part_one(test=False):
    if test:
        data = testdata
    else:
        data = file

    counter = 0
    for line in data:
        splitter = line.split(' | ')
        output = [x for x in splitter[1].split(' ')]

        for o in output:
            if len(o) in [2, 4, 3, 7]:
                counter += 1

    print(f'Unique occurences of digits 1, 4, 7 and 8: {counter}')
    
def part_two(test=False):
    if test:
        data = testdata
    else:
        data = file

    for line in data:
        splitter = line.split(' | ')
        pattern = [x for x in splitter[0].split(' ')]
        output = [x for x in splitter[1].split(' ')]

        numbers = {
            'o': [],
            'lo': [],
            'ro': [],
            'm': [],
            'lu': [],
            'ru': [],
            'u': []
        }
        while True:
            for o in output:
                if len(o) in 2: # 1
                    for x in o:
                        numbers['ro'].append(x)
                        numbers['ru'].append(x)

if __name__ == "__main__":
    part_one()
    part_two(True)
