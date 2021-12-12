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

    total = 0
    for line in data:
        result = 0
        splitter = line.split(' | ')
        pattern = [x for x in splitter[0].split(' ')]
        output = [x for x in splitter[1].split(' ')]

        """ 
        The 7-segment display has 4 unique segments defined by the numbers with unique length.
            a: top-dash
            b: top-right and bottom-right
            c: top-left and middle-dash
            d: bottom-left and bottom-dash
        With those 4 segments the other 6 numbers can be found together with their string-length.
        """
        segments = {'a': [], 'b': [], 'c': [], 'd': []}
        unique_set = 0
        while True:
            # set unique patterns
            for code in pattern:
                if len(code) == 2 and len(segments['b']) == 0:
                    segments['b'] += code
                    unique_set += 1
                elif len(code) == 3 and len(segments['a']) == 0 and len(segments['b']) != 0:
                    for s in code:
                        if s not in segments['b']:
                            segments['a'].append(s)
                    unique_set += 1
                elif len(code) == 4 and len(segments['c']) == 0  and len(segments['b']) != 0:
                    for s in code:
                        if s not in segments['b']:
                            segments['c'].append(s)
                    unique_set += 1
                elif len(code) == 7 and len(segments['a']) != 0 and len(segments['b']) != 0 \
                        and len(segments['c']) != 0 and len(segments['d']) == 0:
                    for s in code:
                        if s not in segments['a'] and s not in segments['b'] and s not in segments['c']:
                            segments['d'].append(s)
                    unique_set += 1

            if unique_set == 4:
                break

        output_number = []
        for o in output:
            if len(o) == 2:
                output_number.append("1")
            elif len(o) == 4:
                output_number.append("4")
            elif len(o) == 3:
                output_number.append("7")
            elif len(o) == 7:
                output_number.append("8")

            elif len(o) == 5:
                # either a 2, 3, or 5
                TWO = 0
                THREE = 0
                FIVE = 0
                for s in o:
                    if s in segments['d']:
                        TWO += 1
                    if s in segments['b']:
                        THREE += 1
                    if s in segments['c']:
                        FIVE += 1

                if TWO == 2:
                    output_number.append("2")
                elif THREE == 2:
                    output_number.append("3")
                elif FIVE == 2:
                    output_number.append("5")
                else:
                    raise Exception("Wrong pattern for length=5: " + o)

            elif len(o) == 6:
                # either a 0, 6, or 9
                ZERO = 0
                SIX = 0
                NINE = 0
                for s in o:
                    if s in segments['b']+segments['d']:
                        ZERO += 1
                    if s in segments['c']+segments['d']:
                        SIX += 1
                    if s in segments['b']+segments['c']:
                        NINE += 1
                    
                if ZERO == 4:
                    output_number.append("0")
                elif SIX == 4:
                    output_number.append("6")
                elif NINE == 4:
                    output_number.append("9")
                else:
                    raise Exception("Wrong pattern for length=6: " + o)
            else:
                raise Exception("Wrong length found! Number: " + o)
        
        result = int(''.join(output_number))
        total += result

    print(f'The output value is {total}.')

if __name__ == "__main__":
    part_one()
    part_two()
