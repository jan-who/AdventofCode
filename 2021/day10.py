from utils import read_input

file = read_input("day10")

testdata = """[({(<(())[]>[[{[]{<()<>>
[(()[<>])]({[<{<<[]>>(
{([(<{}[<>[]}>{[]{[(<()>
(((({<>}<{<{<>}{[]{[]{}
[[<[([]))<([[{}[[()]]]
[{[{({}]{}}([{[{{{}}([]
{<[[]]>}<{[{[{[]{()[[[]
[<(<(<(<{}))><([]([]()
<{([([[(<>()){}]>(<<{{
<{([{{}}[<[[[<>{}]]]>[]]"""
testdata = [x for x in testdata.splitlines()]

def part_one(test=False):
    if test:
        data = testdata
    else:
        data = file
    
    syntax_score = 0
    for line in data:
        openings = []
        for x in line:
            if x in ['[','(','{','<']:
                openings.append(x)
            elif x in [']',')','}','>']:
                last_opening = openings.pop()
                if x == ')' and last_opening != '(':
                    syntax_score += 3
                    break
                elif x == ']' and last_opening != '[':
                    syntax_score += 57
                    break
                elif x == '}' and last_opening != '{':
                    syntax_score += 1197
                    break
                elif x == '>' and last_opening != '<':
                    syntax_score += 25137
                    break
            else:
                raise Exception('Unknown type found: ', x)

    print(f'Syntax score is {syntax_score}')

def part_two(test=False):
    if test:
        data = testdata
    else:
        data = file
    
    noncorrupted_data = []
    for line in data:
        noncorrupt_line = ''
        openings = []
        error = False
        for x in line:
            if x in ['[','(','{','<']:
                openings.append(x)
            elif x in [']',')','}','>']:
                last_opening = openings.pop()
                if x == ')' and last_opening != '(':
                    error = True
                    break
                elif x == ']' and last_opening != '[':
                    error = True
                    break
                elif x == '}' and last_opening != '{':
                    error = True
                    break
                elif x == '>' and last_opening != '<':
                    error = True
                    break
            else:
                raise Exception('Unknown type found: ', x)

            noncorrupt_line += x

        if not error:
            noncorrupted_data.append(noncorrupt_line)

    scores = []
    for line in noncorrupted_data:
        openings = []
        for x in line:
            if x in ['[','(','{','<']:
                openings.append(x)
            elif x in [']',')','}','>']:
                last_opening = openings.pop()
            else:
                raise Exception('Unknown type found: ', x)
        
        score = 0
        openings.reverse()
        for o in openings:
            score *= 5
            if o == '(':
                score += 1
            elif o == '[':
                score += 2
            elif o == '{':
                score += 3
            elif o == '<':
                score += 4
        scores.append(score)
    scores.sort()
    len_scores = len(scores)
    print(f'The middle score is: {scores[len_scores // 2]}')

if __name__ == "__main__":
    part_one()
    part_two()
