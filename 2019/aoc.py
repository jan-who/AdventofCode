from collections import deque

POSITION = 0
IMMEDIATE = 1
RELATIVE = 2

ADD = 1
MUL = 2
INP = 3
OUT = 4
JUMP_TRUE = 5
JUMP_FALSE = 6
LESS_THAN = 7
EQUALS = 8
RELATIVE_BASE = 9
HALT = 99

READ = 0
WRITE = 1

OPS = {
    ADD: (READ, READ, WRITE),
    MUL: (READ, READ, WRITE),
    INP: (WRITE,),
    OUT: (READ,),
    JUMP_TRUE: (READ, READ),
    JUMP_FALSE: (READ, READ),
    LESS_THAN: (READ, READ, WRITE),
    EQUALS: (READ, READ, WRITE),
    RELATIVE_BASE: (READ,),
    HALT: ()
}

class Intcode():    
    """
    Intcode Class: opcode, input=None, debug=False
    """
    def __init__(self, opcode, input=None, debug=False):
        self.debug = debug
        self.memory = opcode
        if self.debug and input: print("Input:", input)
        self.input = deque(input if input else [])
        self.output = []
        if self.debug: print("Initilising machine.")
            
    def __getitem__(self, index):
        return self.memory[index]
    
    def __setitem__(self, index, value):
        self.memory[index] = value
        
    def push_in(self, input):
        if self.debug: print("Input:", input)
        self.input.extend(input)
    
    def pop_out(self):
        if self.debug: print("Output:", self.output)
        output = self.output
        self.output = []
        return output
        
    def get_params(self, param_kinds, parameter):
        args = [None] * 3
        for i, kind in enumerate(param_kinds):
            current_parameter = parameter % 10
            parameter //= 10
            
            arg = self[self.idx + 1 + i]
            
            if kind not in (READ, WRITE):
                raise Exception(f"Invalid kind: {kind}!")
            
            if current_parameter == RELATIVE:
                arg += self.relative_base
                
            if current_parameter in (POSITION, RELATIVE):
                if arg < 0:
                    raise Exception(f"Invalid access to memory at {arg} for index {self.idx}!")
                elif arg >= len(self.memory):
                    #if self.debug: print(f"Extending address to {arg + 1 - len(self.memory)}.")
                    self.memory += [0] * (arg + 1 - len(self.memory))
                
                if kind == READ:
                    arg = self[arg]
                    
            elif current_parameter == IMMEDIATE:
                if kind == WRITE:
                    raise Exception(f"Invalid parameter mode: {current_parameter} at index {self.idx}!")
                
            args[i] = arg
            
        return args
        
    def run(self):
        if self.debug: print("Starting machine.")
        self.idx = 0
        self.relative_base = 0
        
        while self[self.idx] != HALT:
            opcode = self[self.idx]
            instruction = opcode % 100
            parameter = opcode // 100
            
            if instruction not in OPS:
                raise Exception(f"Instruction not found: {instruction}!")
            
            param_kinds = OPS[instruction]
            a, b, c = self.get_params(param_kinds, parameter)            
            #if self.debug: print(f"Index {self.idx}: {opcode} with {a}, {b}, and {c}")
            
            self.idx += 1 + len(param_kinds)
            
            if instruction == INP:
                while not self.input:
                    yield
                self[a] = self.input.popleft()
                
            elif instruction == OUT:
                self.output.append(a)
                
            elif instruction == ADD:
                self[c] = a + b
                
            elif instruction == MUL:
                self[c] = a * b
            
            elif instruction == JUMP_TRUE:
                if a != 0:
                    self.idx = b
            
            elif instruction == JUMP_FALSE:
                if a == 0:
                    self.idx = b
            
            elif instruction == LESS_THAN:
                self[c] = 1 if a < b else 0
                
            elif instruction == EQUALS:
                self[c] = 1 if a == b else 0
            
            elif instruction == RELATIVE_BASE:
                self.relative_base += a
                #if self.debug: print(">>> Setting relative base to:", self.relative_base)
            
            else:
                raise Exception(f"Unimplemented instruction: {opcode}!")
            
        if self.debug: print("Shutting down. ", end='')
        if self.debug and self[self.idx] == HALT: print("OK.")
        if self.debug and self[self.idx] != HALT: print("ERROR.")

def read_input(filename):
    with open(filename, 'r') as file:
        data = file.read().splitlines()
    return data

def position_to_str(x, y):
    str_x = ''
    str_y = ''
    if x >= 0: str_x += '+'
    if y >= 0: str_y += '+'
    str_x += str(x)
    str_y += str(y)
    return str_x.zfill(4) + ',' + str_y.zfill(4)
        
def day_one():
    def get_fuel(mass):
        return mass // 3 - 2
    
    def get_fuel_fuel(mass):
        fuel = get_fuel(mass)
        if fuel <= 0:
            return 0
        fuel += get_fuel_fuel(fuel)
        return fuel
        total_fuel = 0
        
    with open('1_input.txt','r') as file:
        for line in file:
            total_fuel += get_fuel_fuel(int(line))
        
    print(f"Total amount of fuel is: {total_fuel}")
#day_one()

def day_two(noun=12, verb=2, verbose=False):   
    opcodes_orig = [1,0,0,3,1,1,2,3,1,3,4,3,1,5,0,3,2,13,1,19,1,10,19,23,1,6,23,27,1,5,27,31,1,10,31,35,2,10,35,39,1,39,5,43,2,43,6,47,2,9,47,51,1,51,5,55,1,5,55,59,2,10,59,63,1,5,63,67,1,67,10,71,2,6,71,75,2,6,75,79,1,5,79,83,2,6,83,87,2,13,87,91,1,91,6,95,2,13,95,99,1,99,5,103,2,103,10,107,1,9,107,111,1,111,6,115,1,115,2,119,1,119,10,0,99,2,14,0,0]
    opcodes = opcodes_orig.copy()
    
    opcodes[1] = noun
    opcodes[2] = verb

    for i in range(0,len(opcodes),4):
        if verbose: print(f"Now at index {i}: {opcodes[i]}")
        if opcodes[i] == 99:
            if verbose: print("Beende Programm.")
            break 
        elif opcodes[i] == 1:
            if verbose: print(f"Adding {opcodes[opcodes[i+1]]} + {opcodes[opcodes[i+2]]} to index {opcodes[i+3]}")
            opcodes[opcodes[i+3]] = opcodes[opcodes[i+1]] + opcodes[opcodes[i+2]]
        elif opcodes[i] == 2:
            if verbose: print(f"Multiplying {opcodes[i+1]} * {opcodes[i+2]} to {opcodes[i+3]}")
            opcodes[opcodes[i+3]] = opcodes[opcodes[i+1]] * opcodes[opcodes[i+2]]
    
    if verbose: print(f"Value at position 0 after the program halts: {opcodes[0]}")
    if verbose: print(f"100 * noun + verb = {100*noun+verb}")
    return opcodes[0], noun, verb
#print(day_two())
def day_two_part_two(answer):
    res, noun, verb = day_two()
    
    while res != answer:
        verbose = False
        
        # verb für hintere drei Zahlen,
        # noun für vordere Zahlen
        res2 = res % 1000
        res1 = res - res2
        ans2 = answer % 1000
        ans1 = answer - ans2

        if res2 < ans2:
            verb += 1
            if verbose: print(f"Adding +1 to verb to {verb}")
        elif res2 > ans2:
            verb -= 1
            if verbose: print(f"Subtracting +1 to verb to {verb}")
    
        if res1 < ans1:
            noun += 1
            if verbose: print(f"Adding +1 to noun to {noun}")
        elif res1 > ans1:
            noun -= 1
            if verbose: print(f"Subtracting +1 to noun to {noun}")
            
        res, noun, verb = day_two(noun, verb)
    
    print(f"Final value found is {res} with verb {verb} and noun {noun}.")
    print(f"100 * noun + verb is {100*noun+verb}")
#day_two_part_two(19690720)

def day_three(verbose):
    with open('2_input.txt','r') as file:
        inp = file.read().splitlines()
    wa, wb = inp
    
    wa = wa.split(',')
    wb = wb.split(',')

    #wa = ['R8','U5','L5','D3']
    #wb = ['U7','R6','D4','L4']
    #wa = ['R75','D30','R83','U83','L12','D49','R71','U7','L72']
    #wb = ['U62','R66','U55','R34','D71','R55','D58','R83']
    #wa = ['R98','U47','R26','D63','R33','U87','L62','D20','R33','U53','R51']
    #wb = ['U98','R91','D20','R16','D67','R40','U7','R15','U6','R7']

    def add_wire(wire, s, v=False):
        x = 0
        y = 0
        cnt = 0
        
        for seg in wire:
            command = seg[0]
            steps = int(seg[1:])
            
            for i in range(steps):
                cnt += 1
                if command == 'U':
                    y += 1
                elif command == 'D':
                    y -= 1
                elif command == 'L':
                    x -= 1
                elif command == 'R':
                    x += 1
                
                s[(x,y)] = cnt
            
            if v: print(f"seg={seg} at (x,y) = {x,y}")
    
    a = dict()
    b = dict()
    
    add_wire(wa,a,verbose)
    add_wire(wb,b,verbose)    

    both = set(a.keys()).intersection(set(b.keys()))
    if verbose: print(f"Intersection points: {both}")
    
    dist = None
    for pos in both:
        d = abs(pos[0])+abs(pos[1])
        if dist is None or abs(dist[0])+abs(dist[1]) > d:
            dist = pos
    
    answer2 = None
    for point in both:
        step_a = a[point]
        step_b = b[point]
        if verbose: print(f"At intersection {point} we got {step_a} + {step_b} = {step_a+step_b}")
        
        if answer2 is None or answer2 > step_a+step_b:
            answer2 = step_a+step_b
    
    print(f"The shortest intersection can be found at {dist} with distance {abs(dist[0])+abs(dist[1])}.")
    print(f"The fewest steps needed are {answer2}.")
#day_three(False)

def day_four(range_from, range_to, verbose):
    from collections import Counter
    
    total = 0
    for i in range(range_from, range_to+1):
        str_i = str(i)
        sorted_str = ''.join(sorted(str_i))
        
        if str_i == sorted_str: # aufsteigende Reihenfolge
            c = Counter(str_i) # zählt Elemente
            
            ## Part I
            #mc = c.most_common(1) # gibt das häufigste Element zurück
            #max_mc = max(mc)[1] # Konvertierung der Anzahl aus dict zu int
            #if max_mc >= 2:
            #    total += 1
            #    if verbose: print(f"Found {str_i} with number {max(mc)[0]} {max_mc} times!")
             
            ## Part II
            if 2 in c.values(): # c.values() gibt die Häufigkeiten aus dem Counter-dict zurück
                total += 1
                if verbose: print(f"Found {str_i}!")
            
    print(f"Total possible combinations are {total}.")
    return total
#day_four(146810,612564,False)

def day_five(verbose, user_input=[], opcodes=False, verbose2=True, i=0):  
    base = 0
    
    def get_param(instruction, paramno):
        if instruction == 0:
            pos = opcodes[i+paramno]
            try:
                rc = opcodes[pos]
            except:
                if op == 1 or op == 2:
                    opcodes[pos] = 0
                    rc = 0
                else:
                    rc = None
        elif instruction == 1:
            try:
                pos = i+paramno
                rc = opcodes[pos]
            except:
                rc = None
        elif instruction == 2:
            rel_pos = opcodes[i+paramno]
            pos = base + rel_pos
            if verbose: print(f">>> Base is currently {base}, adding {rel_pos} to it. Getting data from index {pos}")
            try:
                rc = opcodes[pos]
            except:
                rc = 0
        else:
            rc = None
            pos = None
        return rc, pos

    if not opcodes:
        print("New opcodes!")
        with open('5_input.txt','r') as file:
            data = file.read()
        #data = "3,21,1008,21,8,20,1005,20,22,107,8,21,20,1006,20,31,1106,0,36,98,0,0,1002,21,125,20,4,20,1105,1,46,104,999,1105,1,46,1101,1000,1,20,4,20,1105,1,46,98,99"
        codes = [int(x) for x in data.split(',')]
        opcodes = dict()
        for a in range(len(codes)):
            opcodes[a] = codes[a]

    ret = []
    while i <= len(opcodes):
        intcode = str(opcodes[i]).zfill(5)
        op = int(intcode[3:])
        p1 = int(intcode[2])
        p2 = int(intcode[1])
        p3 = int(intcode[0])
        
        if verbose: print(f"Now at index {i}: {intcode}")
        
        # Kein Parameter
        if op == 99: # END
            if verbose2: print(">>> Beende Programm.")
            return ret, op, i, opcodes

        # OP PARAM1: IN, OUT, BASE
        elif op == 3 or op == 4 or op == 9:
            # Param1
            param1, pos1 = get_param(p1,1)
            
            if op == 3:
                if len(user_input) > 0:
                    if verbose2: print(f"First input of {user_input} written to index {pos1}")
                    current_input = int(user_input.pop(0))
                    opcodes[pos1] = current_input
                else:
                    if verbose2: print(f">>> Keine Inputs mehr. Beende Programm vorläufig.")
                    if verbose2: print()
                    return ret, op, i, opcodes
            elif op == 4: # OUT
                if verbose2: print(f">>> Diagnostic output: {param1}")
                ret.append(param1)
            elif op == 9: # BASE
                base += param1
                if verbose: print(f"Relative base was {base-param1} and is now at {base}")
            i += 2
            
        # OP PARAM1, PARAM2: CHECK_NOT_NULL, CHECK_IF_NULL
        elif op ==5 or op == 6:
            # Param1, Param2
            param1, _ = get_param(p1,1)
            param2, _ = get_param(p2,2)
            
            if op == 5: # CHECK_NOT_NULL
                if verbose: print(f"Check if param1={param1} is not 0. If yes setting i from {i} to param2={param2}, {i+3} otherwise.", end='')
                if param1:
                    i = param2
                    if verbose: print(f" Yes. i={i}")
                else:
                    i += 3
                    if verbose: print(f" No. i={i}")
            elif op == 6: # CHECK_IF_NULL
                if verbose: print(f"Check if param1={param1} is 0. If yes setting i from {i} to param2={param2}, {i+3} otherwise.", end='')
                if not param1:
                    i = param2
                    if verbose: print(f" Yes. i={i}")
                else:
                    i += 3
                    if verbose: print(f" No. i={i}")
                    
        # OP PARAM1, PARAM2, PARAM3: ADD, MULT, LESS_THAN, IS_EQUAL
        elif op == 1 or op == 2 or op == 7 or op == 8:
            # Param1, Param2, Param3
            param1, _ = get_param(p1,1)
            param2, _ = get_param(p2,2)
            param3, pos3 = get_param(p3,3)
                 
            if op == 1: # ADD
                if verbose: print(f"Adding {param1} + {param2} = {param1+param2} to index {pos3}")
                if p3 == 1:
                    param3 = param1 + param2
                else:
                    opcodes[pos3] = param1 + param2
            elif op == 2: # MULT
                if verbose: print(f"Multiplying {param1} * {param2} = {param1*param2} to index {pos3}")
                if p3 == 1:
                    param3 = param1 * param2 
                else:
                    opcodes[pos3] = param1 * param2
            elif op == 7: # LESS_THAN
                if param1 < param2:
                    if verbose: print(f"param1={param1} is less than param2={param2}. Therefore setting param3={param3} to 1.")
                    opcodes[pos3] = 1
                else:
                    if verbose: print(f"param1={param1} is NOT less than param2={param2}. Therefore setting param3={param3} to 0.")
                    opcodes[pos3] = 0
            elif op == 8: # IS_EQUAL
                if param1 == param2:
                    if verbose: print(f"param1={param1} equals param2={param2}. Therefore setting param3={param3} to 1.")
                    opcodes[pos3] = 1
                else:
                    if verbose: print(f"param1={param1} equals NOT param2={param2}. Therefore setting param3={param3} to 0.")
                    opcodes[pos3] = 0
            i += 4
        else:
            print(f">>> Wrong instruction. {op}")
            return ret, op, i, opcodes
            
    return ret, op, i, opcodes
#day_five(False,[5])

def day_six(verbose, exercise):
    with open('6_input.txt','r') as file:
        data = file.read().splitlines()
    #data = "COM)AAA\nAAA)BBB\nBBB)CCC\nBBB)DDD\nDDD)EEE\nEEE)YOU\nCCC)SAN".split('\n')
    #data = "COM)B\nB)C\nC)D\nD)E\nE)F\nB)G\nG)H\nD)I\nE)J\nJ)K\nK)L\nK)YOU\nI)SAN".split('\n')

    graph = dict()   
    graph['COM'] = None
    for d in data:
        planet, orbit = d.split(')')
        graph[orbit] = planet
    
    def count_orbits(orbit, ct=0, goal='COM'):
        if verbose: print(f"Now looking at orbit from {orbit}, our goal is {goal}. Current count: {ct}")
        if orbit == goal:
            return ct
        else:
            new_orbit = graph[orbit]
            ct +=1
            return count_orbits(new_orbit, ct, goal)

    def graph_to_com(orbit, map=dict()):
        m = map.copy()
        if graph[orbit] is None:
            if verbose: print(f">> End for this orbit. Total distance to COM is {len(map)}")
            return m
        else:
            if verbose: print(f"Now looking at planet {orbit}, which orbits {graph[orbit]}")
            m[orbit] = graph[orbit]
            return graph_to_com(graph[orbit], m)
    
    
    if exercise == 1:
        ct = 0
        for o in graph:
            ct += count_orbits(o)
        print(f"The total count of all direct and indirect orbits is {ct}")
    elif exercise == 2:
        def search_graph(orbit_a, orbit_b):
            a_to_com = graph_to_com(orbit_a)
            b_to_com = graph_to_com(orbit_b)

            intersection = set(a_to_com) & set(b_to_com)
            
            ct_min = None
            for i in intersection:
                ct_you = count_orbits('YOU',goal=i)-1
                ct_san = count_orbits('SAN',goal=i)-1
                ct_total = ct_you+ct_san
                if ct_min is None or ct_total < ct_min:
                    ct_min = ct_total
                    
                if verbose: print(f"We will travel {ct_you} orbits to intersection {i} and further {ct_san} orbits to Santa for a total of {ct_you+ct_san}.")
            return ct_min
            
        c = search_graph('YOU', 'SAN')
        print(f"Minimum travel distance is {c} orbits.")
#day_six(False,exercise=2)

def day_seven(verbose, part=1):
    with open('7_input.txt','r') as file:
            data = file.read()
    #data = "3,15,3,16,1002,16,10,16,1,16,15,15,4,15,99,0,0"
    #data = "3,23,3,24,1002,24,10,24,1002,23,-1,23,101,5,23,23,1,24,23,23,4,23,99,0,0"
    #data = "3,31,3,32,1002,32,10,32,1001,31,-2,31,1007,31,0,33,1002,33,7,33,1,33,31,31,1,32,31,31,4,31,99,0,0,0"
    
    #data = "3,26,1001,26,-4,26,3,27,1002,27,2,27,1,27,26,27,4,27,1001,28,-1,28,1005,28,6,99,0,0,5"
    #data = "3,52,1001,52,-5,52,3,53,1,52,56,54,1007,54,5,55,1005,55,26,1001,54,-5,54,1105,1,12,1,53,54,53,1008,54,0,55,1001,55,1,55,2,53,55,53,4,53,1001,56,-1,56,1005,56,6,99,0,0,0,0,10"
    codes = [int(i) for i in data.split(',')]
    opcodes = dict()
    for a in range(len(codes)):
        opcodes[a] = codes[a]
            
    if verbose: print(f"OPCODES: {len(opcodes)} with {opcodes}")
    
    max_no_amps = 5    
    if part == 1:
        min_no = 1234
        max_no = 44444
    else:
        min_no = 56789
        max_no = 99999
    
    max_output = None
    best_setting = list()
    if verbose: settings = 0
    for p in range(min_no, max_no):
        phase_setting = list()
        for s in str(p).zfill(max_no_amps):
            s = int(s)
            if s not in phase_setting and ((part == 1 and s <= 4) or (part == 2 and s > 4)):
                phase_setting.append(s)
        if len(phase_setting) < max_no_amps:
            continue

        if verbose: 
            settings += 1
            print("")
            print(f"Current phase setting {phase_setting}")
        amp = [0]
        if part == 1:
            idx = 0
            for i in range(max_no_amps):
                amp, _, idx = day_five(verbose, [phase_setting[i], amp[0]].copy(), opcodes.copy(), idx)
                if verbose: print(f"At AMP {i} with output {amp[0]}.")
            if max_output is None or max_output < amp[0]:
                max_output = amp[0]
                best_setting = phase_setting
        else:
            amps = dict()
            for i in range(max_no_amps):
                amps[i] = {'rc': None, 'inputs': [phase_setting[i]], 'idx': 0}
            if verbose: print(f"Starting with the following setting: {amps}")

            while amps[4]['rc'] != 99:
                for i in range(max_no_amps):
                    curr_idx = amps[i]['idx']
                    
                    if i == 0 and len(amps[i]['inputs']) == 1:
                        amps[i]['inputs'].append(0)
                    
                    if curr_idx == 0:
                        curr_input = amps[i]['inputs'][-2:]
                    else:
                        curr_input = [amps[i]['inputs'][-1]]
                    if verbose: print(f"Current input: {curr_input} at index {curr_idx}.")   
                    
                    amp, rc, idx = day_five(False, curr_input, opcodes.copy(), False, curr_idx) 
                    
                    if verbose: print(f"AMP {i}: With inputs {amps[i]['inputs']} we got the following output: {amp[-1]}.")
                    a = (i+1) % 5
                    amps[i]['rc'] = rc
                    amps[i]['idx'] = idx
                    amps[a]['inputs'].append(amp[-1])
            if verbose: print(f">>> Now it is: {amps}")
            if max_output is None or max_output < amp[-1]:
                max_output = amp[-1]
                best_setting = phase_setting
    if verbose: print("")
    if verbose: print(f"Found {settings} different phase setting.")
    print("")         
    print(f"The output with phase setting {best_setting} is {max_output}.")
#day_seven(False,2)

def day_eight(verbose, width, height, part=1):
    from collections import Counter
    with open('8_input.txt','r') as file:
        data = file.read().strip()
    #data = "100456789012"
    #data = "0222112222120000"
    pixels_per_layer = width * height
    num_layers = len(data) // pixels_per_layer
    if verbose: print(f"The image contains {len(data)} pixels and is a {width}x{height} image, which gives us {num_layers} layers.")
    layers = dict()
    for i in range(num_layers):
        layers[i] = list()
    for i, d in enumerate(data):
        l = i // pixels_per_layer
        layers[l].append(int(d))
    if verbose: print(f"The layers: {layers}")
    if part == 1:
        layer_fewest_zeros = None
        c_fewest_zeros = None
        for i in range(num_layers):
            c = Counter(layers[i])[0]
            if verbose: print(f"Layer {i} has {c} 0s in it!")
            if c_fewest_zeros is None or c_fewest_zeros > c:
                layer_fewest_zeros = i
                c_fewest_zeros = c
        print(f"The layer with the fewest zeros is layer {layer_fewest_zeros} with {c_fewest_zeros} occurences.")
        c_1 = Counter(layers[layer_fewest_zeros])[1]
        c_2 = Counter(layers[layer_fewest_zeros])[2]
        print(f"The layer {layer_fewest_zeros} has {c_1} ones and {c_2} twos, so the answer is: {c_1*c_2}!")
    elif part == 2:
        img = ""
        for pos in range(pixels_per_layer):
            pixel = None
            for l in range(num_layers):
                if verbose: print(f"Layer {l}, position {pos}: {layers[l][pos]}")
                if pixel is None or (pixel == 2 and layers[l][pos] != 2):
                    pixel = layers[l][pos]
            img += str(pixel)
        print("The final picture looks like this:")
        for idx, s in enumerate(img):
            if idx % width == 0:
               print("")
            if int(s) == 0:
                p = ' '
            elif int(s) == 1:
                p = '*'
            print(p, end='')
        print("")
#day_eight(False,25,6,2)

def day_nine(verbose, inputs=False, verbose2=False):
    with open('9_input.txt','r') as file:
        data = file.read().strip()
    #data = "109,1,204,-1,1001,100,1,100,1008,100,16,101,1006,101,0,99"
    #data = "1102,34915192,34915192,7,4,7,99,0"
    #data = "104,1125899906842624,99"
    #data = "109,2000,109,19,204,-2017,99"
    codes = [int(i) for i in data.split(',')]
    opcodes = dict()
    for a in range(len(codes)):
        opcodes[a] = codes[a]
    if verbose: print(f"OPCODES: {len(opcodes)} with {opcodes}")
    code, _, _ = day_five(verbose, inputs, opcodes.copy(), verbose2)
    print(f"The returncode is: {code[0]}")
#day_nine(False, [2])

def day_ten(verbose, part, laser=False):
    from collections import Counter
    import math
    
    def look_for_asteroid(x, y, a_x, a_y, debug=True):
        if x < 0 or x >= width or y < 0 or y >= height or (x == a_x and y == a_y):
            return False
        if debug: print(f"Asteroid at {str(a_x).zfill(2)+str(a_y).zfill(2)}, now looking at {str(x).zfill(2)+str(y).zfill(2)}...", end='')
        point = map[y][x]
        if point == '#':
            if debug: print(f" found an asteroid!")
            if part == 1:
                asteroid_map[coords] += 1
            else:
                possible_asteroids[(str(x).zfill(2)+str(y).zfill(2))] = None
            return True
        else:
            if debug: print("")
            return False
            
    with open('10_input.txt','r') as file:
        data = file.read()
    #data = ".#..#\n.....\n#####\n....#\n...##"
    #data = "......#.#.\n#..#.#....\n..#######.\n.#.#.###..\n.#..#.....\n..#....#.#\n#..#....#.\n.##.#..###\n##...#..#.\n.#....####"
    #data = "#.#...#.#.\n.###....#.\n.#....#...\n##.#.#.#.#\n....#.#.#.\n.##..###.#\n..#...##..\n..##....##\n......#...\n.####.###."
    #data = ".#..#..###\n####.###.#\n....###.#.\n..###.##.#\n##.##.#.#.\n....###..#\n..#.#..#.#\n#..#.#.###\n.##...##.#\n.....#.#.."
    #data = ".#..##.###...#######\n##.############..##.\n.#.######.########.#\n.###.#######.####.#.\n#####.##.#.##.###.##\n..#####..#.#########\n####################\n#.####....###.#.#.##\n##.#################\n#####.##.###..####..\n..######..##.#######\n####.##.####...##..#\n.#####..#.######.###\n##...#.##########...\n#.##########.#######\n.####.#.###.###.#.##\n....##.##.###..#####\n.#.#.###########.###\n#.#.#.#####.####.###\n###.##.####.##.#..##"
    #data = ".#....#####...#..\n##...##.#####..##\n##...#...#.#####.\n..#.....X...###..\n..#.#.....#....##"
    
    map = [[s for s in line] for line in data.split("\n")]
    c_asteroids = Counter(data)['#']
    width = len(map[0])
    height = len(map)
    if verbose: print(f"Map of {width}x{height} with {c_asteroids} asteroids.")
    
    factors = ['0001','0100','0101']
    factor_mults = []
    for w in range(1,width):
        for h in range(1,height):
            if w != h and w/h not in factor_mults:
                factors.append(str(w).zfill(2)+str(h).zfill(2))
                factor_mults.append(w/h)
    if verbose: print(f"Factors: {factors}")
    
    asteroid_map = dict()
    asteroid_list = list()
    for c in range(c_asteroids):
        for x in range(width):
            for y in range(height):
                coordinates = str(x).zfill(2)+str(y).zfill(2)
                point = map[y][x]
                if (point == '#' or point == 'X') and coordinates not in asteroid_map:
                    asteroid_map[coordinates] = 0
                    asteroid_list.append(coordinates)

    if part == 1:
        for coords in asteroid_list:
            a_x = int(coords[:2])
            a_y = int(coords[2:])
            if a_x is None or a_y is None:
                break
    
            for f in factors:
                f_x = int(f[:2])
                f_y = int(f[2:])
                if verbose: print(f"Current factor: {f}")
                
                i = 1
                x_pos_y_pos = False
                x_neg_y_pos = False
                x_pos_y_neg = False
                x_neg_y_neg = False
                while True:
                    x = f_x*i
                    y = f_y*i
                    
                    if not x_pos_y_pos:
                        x_pos_y_pos = look_for_asteroid(a_x+x, a_y+y, a_x, a_y, verbose)
                    if not x_neg_y_pos and x != 0:
                        x_neg_y_pos = look_for_asteroid(a_x-x, a_y+y, a_x, a_y, verbose)
                    if not x_pos_y_neg and y != 0:
                        x_pos_y_neg = look_for_asteroid(a_x+x, a_y-y, a_x, a_y, verbose)
                    if not x_neg_y_neg and x != 0 and y != 0:
                        x_neg_y_neg = look_for_asteroid(a_x-x, a_y-y, a_x, a_y, verbose)
                    i += 1
                    if x >= width or y >= height:
                        break
            if verbose: print("")
        if verbose: print(f"Asteroid count: {asteroid_map}")
        c = max(asteroid_map, key=asteroid_map.get)
        print(f"Best asteroid is at {int(c[:2])},{int(c[2:])} with {asteroid_map[c]} asteroids detected!")
    elif part == 2:
        if not laser:
            return
        coords = laser
        a_x = int(coords[:2])
        a_y = int(coords[2:])
        map[a_y][a_x] = 'X'
        
        for k in asteroid_map:
            asteroid_map[k] = None
        
        sorted_asteroids = list()
        currrent_count_asteroids = c_asteroids
        breakup = 0
        while currrent_count_asteroids > 1:
            currrent_count_asteroids = 0
            for i in range(len(map)):
                currrent_count_asteroids += Counter(map[i])['#']
            if verbose: print(f"{currrent_count_asteroids} asteroids left.")
            possible_asteroids = dict()
            for f in factors:
                f_x = int(f[:2])
                f_y = int(f[2:])
                
                i = 1
                x_pos_y_pos = False
                x_neg_y_pos = False
                x_pos_y_neg = False
                x_neg_y_neg = False
                while True:
                    x = f_x*i
                    y = f_y*i
                    
                    if not x_pos_y_pos:
                        x_pos_y_pos = look_for_asteroid(a_x+x, a_y+y, a_x, a_y, False)
                    if not x_neg_y_pos and x != 0:
                        x_neg_y_pos = look_for_asteroid(a_x-x, a_y+y, a_x, a_y, False)
                    if not x_pos_y_neg and y != 0:
                        x_pos_y_neg = look_for_asteroid(a_x+x, a_y-y, a_x, a_y, False)
                    if not x_neg_y_neg and x != 0 and y != 0:
                        x_neg_y_neg = look_for_asteroid(a_x-x, a_y-y, a_x, a_y, False)
                        
                    i += 1
                    if x >= width or y >= height:
                        break
            
            for obj in possible_asteroids:
                obj_x = int(obj[:2])
                obj_y = int(obj[2:])
                angle = math.pi - math.atan2(obj_x-a_x, obj_y-a_y)
                possible_asteroids[obj] = angle
            if verbose: print(f"Possible asteroids: {possible_asteroids}")
            if verbose: print()
            
            for k,v in sorted(possible_asteroids.items(), key=lambda item: item[1]):
                sorted_asteroids.append(k)
                x = int(k[:2])
                y = int(k[2:])
                map[y][x] = '_'
                
            #break
            
        for i, coords in enumerate(sorted_asteroids):
            asteroid_map[coords] = i
   
        if verbose: print(f"Final map {asteroid_map}")
        print()
        for num in [1,2,3,10,20,50,100,199,200,201,299]:
            for k, v in asteroid_map.items():
                if v == num-1:
                    x = int(k[:2])
                    y = int(k[2:])
                    print(f"The {num}. asteroid is {x},{y}!")
                    if num == 200:
                        print(f"Answer: {x*100+y}")
                    continue
#day_ten(False,2,laser="1111")

def day_eleven(verbose, part):
    def pos_to_str(x, y):
        str_x = ""
        str_y = ""
        if x >= 0: str_x += "+"
        if y >= 0: str_y += "+"
        str_x += str(x)
        str_y += str(y)
        return str_x.zfill(3) + "," + str_y.zfill(3)
        
    with open('11_input.txt','r') as file:
        data = file.read()
    opcodes = [int(i) for i in data.split(',')]
    
    if part == 1:
        color_current_field = 0
    else:
        color_current_field = 1
        
    breaker = 0
    visited_positions = list()
    direction = '^'
    idx = 0
    current_position = [0,0]
    str_curr_pos = pos_to_str(current_position[0], current_position[1])
    fields = dict()
    fields[str_curr_pos] = color_current_field
    
    vm = Intcode(opcodes, None, verbose)
    machine = vm.run()
    while True:
        vm.push_in([color_current_field])
        visited_positions.append(current_position)
        
        try:
            next(machine)
        except StopIteration:
            if verbose: print("Ende.")
            break
        outputs = vm.pop_out()
        
        paint_current_field = outputs[0]
        turning_direction = outputs[1]
        
        # Farbe auf jetzigem Feld aktualisieren
        fields[str_curr_pos] = paint_current_field
        
        # aufs nächste Feld bewegen
        new_position = current_position.copy()
        # turning_direction=0: 90° links; turning_direction=1: 90° rechts
        if (turning_direction == 0 and direction == '^') or (turning_direction == 1 and direction == 'v'):
            new_position[0] -= 1
            direction = '<'
        elif (turning_direction == 0 and direction == 'v') or (turning_direction == 1 and direction == '^'):
            new_position[0] += 1
            direction = '>'
        elif (turning_direction == 0 and direction == '<') or (turning_direction == 1 and direction == '>'):
            new_position[1] += 1
            direction = 'v'
        elif (turning_direction == 0 and direction == '>') or (turning_direction == 1 and direction == '<'):
            new_position[1] -= 1
            direction = '^'
        # Bewegung vollziehen
        str_curr_pos = pos_to_str(new_position[0], new_position[1])
        if str_curr_pos not in fields:
            fields[str_curr_pos] = 0 # alle Felder sind schwarz
        
        if verbose: 
            if paint_current_field == 1: str_col = 'white'
            else: str_col = 'black'
            if turning_direction == 0: str_turn = 'left'
            else: str_turn = 'right'
            print(f"Field {current_position} which was {color_current_field} and will be painted ", end='')
            print(f"{str_col} ({paint_current_field}), then turning {str_turn} ({turning_direction}).")

        current_position = new_position.copy()    
        color_current_field = fields[str_curr_pos] 
          
        breaker += 1
        if verbose and breaker % 20 == 0: print()
        if verbose and breaker == 200000: 
            print(">>> BREAKING")
            break
    """
    while True:
        visited_positions.append(current_position)
        
        outputs, rc, new_idx, opcodes = day_five(False, [color_current_field], opcodes, False, idx)
        
        paint_current_field = outputs[0]
        turning_direction = outputs[1]
        
        # Farbe auf jetzigem Feld aktualisieren
        fields[str_curr_pos] = paint_current_field
        
        # aufs nächste Feld bewegen
        new_position = current_position.copy()
        # turning_direction=0: 90° links; turning_direction=1: 90° rechts
        if (turning_direction == 0 and direction == '^') or (turning_direction == 1 and direction == 'v'):
            new_position[0] -= 1
            direction = '<'
        elif (turning_direction == 0 and direction == 'v') or (turning_direction == 1 and direction == '^'):
            new_position[0] += 1
            direction = '>'
        elif (turning_direction == 0 and direction == '<') or (turning_direction == 1 and direction == '>'):
            new_position[1] += 1
            direction = 'v'
        elif (turning_direction == 0 and direction == '>') or (turning_direction == 1 and direction == '<'):
            new_position[1] -= 1
            direction = '^'
        # Bewegung vollziehen
        str_curr_pos = pos_to_str(new_position[0], new_position[1])
        if str_curr_pos not in fields:
            fields[str_curr_pos] = 0 # alle Felder sind schwarz
        
        if verbose: 
            if paint_current_field == 1: str_col = 'white'
            else: str_col = 'black'
            if turning_direction == 0: str_turn = 'left'
            else: str_turn = 'right'
            print(f"{idx}: Field {current_position} which was {color_current_field} and will be painted", end='')
            print(f"{str_col} ({paint_current_field}), then turning {str_turn} ({turning_direction}) to field ", end='')
            print(f"{new_position} and index {new_idx}.")

        current_position = new_position.copy()    
        color_current_field = fields[str_curr_pos] 
        idx = new_idx
        
        if rc == 99:
            visited_positions.append(current_position)
            break
            
        breaker += 1
        if verbose and breaker % 20 == 0: print()
        if verbose and breaker == 200000: 
            print(">>> BREAKING")
            break
    """
    min_x = None
    max_x = None
    min_y = None
    max_y = None
    for p in visited_positions:
        if min_x is None or min_x > p[0]:
            min_x = p[0]
        if max_x is None or max_x < p[0]:
            max_x = p[0]
        if min_y is None or min_y > p[1]:
            min_y = p[1]
        if max_y is None or max_y < p[1]:
            max_y = p[1]
    print(f"Picture of size {max_x-min_x+1}x{max_y-min_y+1}.")

    for h in range(min_y, max_y+1):
        print(f"{str(h).zfill(3)}:  ", end='')
        for w in range(min_x, max_x+1):
            pos = pos_to_str(w, h)
            if pos in fields and fields[pos] == 1:
                print('#', end='')
            else:
                print(' ', end='')
        print("")
        if h == 10: break
        
    if part == 1:
        print(f"Panels painted: {len(fields)}")
#day_eleven(False,2)

def day_twelve(verbose, part):
    import numpy as np
    data = "<x=-6, y=-5, z=-8>\n<x=0, y=-3, z=-13>\n<x=-15, y=10, z=-11>\n<x=-3, y=-8, z=3>"
    #data = "<x=-1, y=0, z=2>\n<x=2, y=-10, z=-7>\n<x=4, y=-8, z=8>\n<x=3, y=5, z=-1>"
    #data = "<x=-8, y=-10, z=0>\n<x=5, y=5, z=10>\n<x=2, y=-7, z=3>\n<x=9, y=-8, z=-3>"
    moons = list()
    for i, line in enumerate(data.split('\n')):
        val_list = list()
        for val in line.split(', '):
            m = val.replace('<','').replace('>','').split('=')
            val_list.append(int(m[1]))
        moons.append(val_list)
    if verbose: print("Moon list:", moons)
    
    def run_steps(position, velocity, steps=float('inf')):
        original_position, original_velocity = position[:], velocity[:]
        step_number = 0
        while step_number < steps and (not step_number or original_position != position or original_velocity != velocity): # time steps
            if verbose: print()
            if verbose: print(f"After {step_number} steps:")
            if verbose: print("Positions:", position)
            if verbose: print("Velocity:", velocity)
            
            p1 = 0
            p2 = 1
            start = 0
            for _ in range(6): # n! / (k!*(n-k)!) => n=4, k=2 ===> 24/4 = 6
                velocity[p1] += 1 if position[p1] < position[p2] else 0
                velocity[p1] -= 1 if position[p1] > position[p2] else 0
                
                velocity[p2] -= 1 if position[p1] < position[p2] else 0
                velocity[p2] += 1 if position[p1] > position[p2] else 0
                p2 += 1
                if p2 == len(position):
                    p1 += 1
                    p2 = p1+1
                    
            for i in range(len(position)):
                position[i] += velocity[i]

            step_number += 1
            #if step_number == 3000: break
        if verbose: print()    
        if verbose: print(f"After {step_number} steps:")
        if verbose: print("Positions:", position)
        if verbose: print("Velocity:", velocity)
        return step_number
    
    px, vx = [x for x, _, _ in moons], [0] * len(moons)
    py, vy = [y for _, y, _ in moons], [0] * len(moons)
    pz, vz = [z for _, _, z in moons], [0] * len(moons)    
    if part == 1:
        for p, v in zip((px, py, pz), (vx, vy, vz)):
            run_steps(p, v, 1000)
        # energy
        total = 0
        total = sum((abs(px[i]) + abs(py[i]) + abs(pz[i])) * (abs(vx[i]) + abs(vy[i]) + abs(vz[i])) for i in range(len(moons)))
        print(f"The total energy for all moons is {total}.")
    elif part == 2:
        steps_x = run_steps(px, vx)
        steps_y = run_steps(py, vy)
        steps_z = run_steps(pz, vz)
        ct = np.lcm(np.lcm(steps_x, steps_y), steps_z)
        print("The total number of steps is", ct)
#day_twelve(False, 2)

def day_thirteen(verbose, part, watch=True):
    import time 
    
    with open('13_input.txt','r') as file:
        data = file.read()
    opcodes = [int(i) for i in data.split(',')]
    score = 0
    
    def build_screen_dict(board):
        x_min = None
        x_max = None
        y_min = None
        y_max = None
        
        for instruction in board:
            x = instruction[0]
            y = instruction[1]
            t = instruction[2]
            
            position = position_to_str(x, y)
            screen[position] = t
            
            if t == 4:
                ball_on = x
            elif t == 3:
                paddle_on = x
            
            if x_max is None or x > x_max: x_max = x 
            if x_min is None or x < x_min: x_min = x 
            if y_min is None or y < y_min: y_min = y 
            if y_max is None or y > y_max: y_max = y 
    
        return x_min, x_max, y_min, y_max, ball_on, paddle_on
    
    def print_board(screen):
        for h in range(y_min, y_max+1):
            for w in range(x_min, x_max+1):
                position = position_to_str(w, h)
                if position in screen:
                    tile = screen[position]
                    if h == 0 and w == -1:
                        score = tile
                        print("SCORE:", score)
                        if not watch: return score
                        continue
                    if watch:                    
                        if tile == 1:
                            print('$', end='') # wall
                        elif tile == 2:
                            print('#', end='') # block
                        elif tile == 3:
                            print('=', end='') # paddle
                        elif tile == 4:
                            print('0', end='') # ball
                        else:
                            print(' ', end='')
                else:
                    if watch: print(' ', end='')
            if watch: print("")
        return score
    screen = dict()
    board = []
    count_block_tiles = 0
    
    if part == 2:
        opcodes[0] = 2
    
    vm = Intcode(opcodes, None, verbose)
    machine = vm.run()   
    next(machine)
    
    while True:
        output = vm.pop_out()
        for i in range(0, len(output), 3):
            instructions = [None] * 3
            instructions[0] = output[i]
            instructions[1] = output[i+1]
            instructions[2] = output[i+2]
            board.append(instructions)
            
            if part == 1 and instructions[2] == 2:
                count_block_tiles += 1
        x_min, x_max, y_min, y_max, ball_on, paddle_on = build_screen_dict(board)
        
        score = print_board(screen)
        if watch: time.sleep(.05) # lets watch the game
            
        if ball_on > paddle_on:
            input = 1
        elif ball_on < paddle_on:
            input = -1
        else:
            input = 0
        vm.push_in([input])
        
        try: 
            next(machine)
        except StopIteration:
            break
        
    if part ==1: 
        print(f"There are {count_block_tiles} block tiles in this board.")
    elif part == 2:
        watch = True
        output = vm.pop_out()
        for i in range(0, len(output), 3):
            instructions = [None] * 3
            instructions[0] = output[i]
            instructions[1] = output[i+1]
            instructions[2] = output[i+2]
            board.append(instructions)
        x_min, x_max, y_min, y_max, ball_on, paddle_on = build_screen_dict(board)
        score = print_board(screen)
        print("The final score is", score)
#day_thirteen(False, 2, False)

def day_fourteen(verbose, part):
    import math
    with open('14_input.txt','r') as file:
        data = file.read().splitlines()
#    data = """10 ORE => 10 A
#1 ORE => 1 B
#7 A, 1 B => 1 C
#7 A, 1 C => 1 D
#7 A, 1 D => 1 E
#7 A, 1 E => 1 FUEL""".split('\n')
#    data = """9 ORE => 2 A
#8 ORE => 3 B
#7 ORE => 5 C
#3 A, 4 B => 1 AB
#5 B, 7 C => 1 BC
#4 C, 1 A => 1 CA
#2 AB, 3 BC, 4 CA => 1 FUEL""".split('\n')
#    data = """157 ORE => 5 NZVS
#165 ORE => 6 DCFZ
#44 XJWVT, 5 KHKGT, 1 QDVJ, 29 NZVS, 9 GPVTF, 48 HKGWZ => 1 FUEL
#12 HKGWZ, 1 GPVTF, 8 PSHF => 9 QDVJ
#179 ORE => 7 PSHF
#177 ORE => 5 HKGWZ
#7 DCFZ, 7 PSHF => 2 XJWVT
#165 ORE => 2 GPVTF
#3 DCFZ, 7 NZVS, 5 HKGWZ, 10 PSHF => 8 KHKGT""".split('\n')
#    data = """2 VPVL, 7 FWMGM, 2 CXFTF, 11 MNCFX => 1 STKFG
#17 NVRVD, 3 JNWZP => 8 VPVL
#53 STKFG, 6 MNCFX, 46 VJHF, 81 HVMC, 68 CXFTF, 25 GNMV => 1 FUEL
#22 VJHF, 37 MNCFX => 5 FWMGM
#139 ORE => 4 NVRVD
#144 ORE => 7 JNWZP
#5 MNCFX, 7 RFSQX, 2 FWMGM, 2 VPVL, 19 CXFTF => 3 HVMC
#5 VJHF, 7 MNCFX, 9 VPVL, 37 CXFTF => 6 GNMV
#145 ORE => 6 MNCFX
#1 NVRVD => 8 CXFTF
#1 VJHF, 6 MNCFX => 4 RFSQX
#176 ORE => 6 VJHF""".split('\n')
#    data = """171 ORE => 8 CNZTR
#7 ZLQW, 3 BMBT, 9 XCVML, 26 XMNCP, 1 WPTQ, 2 MZWV, 1 RJRHP => 4 PLWSL
#114 ORE => 4 BHXH
#14 VRPVC => 6 BMBT
#6 BHXH, 18 KTJDG, 12 WPTQ, 7 PLWSL, 31 FHTLT, 37 ZDVW => 1 FUEL
#6 WPTQ, 2 BMBT, 8 ZLQW, 18 KTJDG, 1 XMNCP, 6 MZWV, 1 RJRHP => 6 FHTLT
#15 XDBXC, 2 LTCX, 1 VRPVC => 6 ZLQW
#13 WPTQ, 10 LTCX, 3 RJRHP, 14 XMNCP, 2 MZWV, 1 ZLQW => 1 ZDVW
#5 BMBT => 4 WPTQ
#189 ORE => 9 KTJDG
#1 MZWV, 17 XDBXC, 3 XCVML => 2 XMNCP
#12 VRPVC, 27 CNZTR => 2 XDBXC
#15 KTJDG, 12 BHXH => 5 XCVML
#3 BHXH, 2 VRPVC => 7 MZWV
#121 ORE => 7 VRPVC
#7 XCVML => 6 RJRHP
#5 BHXH, 4 VRPVC => 5 LTCX""".split('\n')
    l = list()
    r = list()
    for row in data:
        tmp = row.split(' => ')
        rec = tmp[1].split(' ')
        r.append({rec[1]: int(rec[0])})
        l_dict = dict()
        for tup in tmp[0].split(', '):
            a, b = tup.split(' ')
            l_dict[b] = int(a)
        l.append(l_dict)

    def produce_ingridient(ing, amount=1):
        ing_dict = cookbook[ing]
        cycles = math.ceil((amount-in_store[ing]) / ing_dict['yields'])
        if verbose: print(f"PRODUCE: Now producing {ing} {cycles} times, yielding {ing_dict['yields']} per round.")
        for _ in range(cycles):
            in_store[ing] += ing_dict['yields']
            if verbose: print(f"Adding {ing_dict['yields']} {ing} to storage. Now we have {in_store[ing]}.")
            for vals in ing_dict['ingridients'].values():
                next_ing = list(vals)[0]
                needed_amount = int(vals[next_ing])
                get_ingridient(next_ing, needed_amount)
        
    def get_ingridient(ing, amount=1):
        if verbose: print(f"GET: Getting {ing}, needing {amount}. ", end='')
        if ing == 'ORE':
            in_store['ORE'] += amount
            if verbose: print(f"Collecting {amount} ore. Now we have {in_store['ORE']}.")
        else:
            while in_store[ing] < amount:
                if verbose: print(f"In storage is only {in_store[ing]}, we need {amount}.")
                produce_ingridient(ing, amount)
            in_store[ing] -= amount
            if verbose: print(f"Removing {amount} {ing} from storage. {in_store[ing]} left.")
        return in_store['ORE']

    def recipe(element='ORE', amount=1):
        if element == 'FUEL': return amount
        
        ore = 0
        for idx, rec in enumerate(l):
            if element in rec:
                parent = list(r[idx])[0]
                parent_qty = r[idx][parent]
                
                ore += math.ceil(recipe(parent, amount) / parent_qty) * rec[element]
        return ore
        
    if part == 1:
        amount = recipe()
        if verbose: print(); print(in_store); print()
        print(f"We need {amount} ore.")
    elif part == 2:
        ORE_MAX_STORAGE = 1_000_000_000_000
        fuel_min, fuel_max = 1, 100_000_000
        while (fuel_max - fuel_min) > 1:
            m = (fuel_min + fuel_max) // 2
            print("New m!", m)
            if recipe(amount=m) <= ORE_MAX_STORAGE:
                fuel_min = m 
            else:
                fuel_max = m 
 
        print(f"1T ORE ==> {fuel_min} FUEL")
#day_fourteen(False, 2)

def day_fiveteen(verbose, part, watch=False):    
    import time
    from collections import deque

    def print_board(points):
        for h in range(limits[2], limits[3]+1):
            print(' ', end='')
            for w in range(limits[0], limits[1]+1):
                position = (w, h)
                if position in points:
                    tile = points[position]                   
                    if (w,h) == (0,0) and part == 1:
                        print('X', end='') # Start
                    elif tile == 2:
                        print('O', end='') # Goal
                    elif (w,h) == (x,y):
                        print('D', end='') # droid
                    elif tile == 0:
                        print('#', end='') # wall
                    elif tile == 1:
                        print('.', end='') # block
                    else:
                        print(' ', end='')
                else:
                    print(' ', end='')
            print("")

    data = read_input('15_input.txt')
    opcodes = [int(i) for i in data[0].split(',')]
    
    if part == 1:
        limits = [0] * 4
        vm = Intcode(opcodes, None, False)
        points = dict()
        points[(0,0)] = 1
        visited = deque([[1],[2],[3],[4]])
        goal_routes = []
        while True:       
            if len(visited) == 0: break
    
            if watch: print(chr(27) + "[2J")
            if watch: print_board(points)
            # prüfe jede Richtung von der aktuellen Position
            movement = visited.popleft()
            vm.push_in(movement)
    
            try:
                next(vm.run())
            except StopIteration:
                break
            out = vm.pop_out()
            
            x, y = 0, 0
            for m in movement:
                if m == 1: y += 1
                elif m == 2: y -= 1
                elif m == 3: x -= 1
                elif m == 4: x += 1
            limits[0] = min(limits[0], x)
            limits[1] = max(limits[1], x)
            limits[2] = min(limits[2], y)
            limits[3] = max(limits[3], y)
            points[(x,y)] = out[-1]
            
            if verbose: print("Tried", movement, end='')
            if out[-1] != 0:
                if verbose: print(" and found something.")
                if out[-1] == 2:
                    print("Found the goal!")
                    goal_routes.append(movement)
                else:    
                    for i in range(1,5):
                        if movement[-1] + i in (3,7):
                            continue
                        next_point = movement + [i]
                        if verbose: print("Adding", next_point, "to queue")
                        visited.append(next_point)
                movement.reverse()
            else:
                if verbose: print(" and ran into a wall.")
                movement.reverse()
                movement = movement[1:]
            # go back to starting point:
            if verbose: print("Let's go back to the starting point!")
            go_back = []
            for p in movement:
                if p == 1: go_back.append(2)
                if p == 2: go_back.append(1)
                if p == 3: go_back.append(4)
                if p == 4: go_back.append(3)
            if len(go_back) > 0:
                vm.push_in(go_back)
                next(vm.run())
                if verbose: print(go_back, "=> I'm back!", vm.pop_out())
            if watch: time.sleep(.13)
            
        print("Goal found at:", goal_routes)
        print([len(g) for g in goal_routes])
        print_board(points)
        
        with open('15_data.txt', 'w') as file:
            file.write(str(points))
        with open('15_data2.txt', 'w') as file:
            file.write(str(goal_routes))    
    elif part == 2:
        limits = [-21,21,-21,21]
        x, y = None, None
        points = ""
        with open('15_data.txt', 'r') as file:
            for i in file.readlines():
                points += i
        points = eval(points)
        goal_route = ""
        with open('15_data2.txt', 'r') as file:
            for i in file.readlines():
                goal_route += i
        goal_route = eval(goal_route)[0]
        goal_route.reverse()
        for key, value in points.items():
            if value == 2:
                starting_point = key
        
        x, y = 0, 0
        ticks = 0
        oxygen = deque([[starting_point]])
        while True:
            if len(oxygen[0]) == 0: break
            current_path = oxygen.popleft()
            new = []
            for p in current_path:
                xn, yn = p
                for _ in range(4):
                    if points[(xn+1,yn)] == 1: 
                        points[(xn+1,yn)] = 2
                        new.append((xn+1,yn))
                    if points[(xn-1,yn)] == 1: 
                        points[(xn-1,yn)] = 2
                        new.append((xn-1,yn))
                    if points[(xn,yn+1)] == 1: 
                        points[(xn,yn+1)] = 2
                        new.append((xn,yn+1))
                    if points[(xn,yn-1)] == 1: 
                        points[(xn,yn-1)] = 2
                        new.append((xn,yn-1))
            oxygen.append(new)
            ticks += 1
            if watch: print(chr(27) + "[2J")
            if watch: print("Ticks:", ticks)
            if watch: print_board(points)
            if watch: time.sleep(.1)
        ticks -= 1
        print_board(points)
        print("Used time:", ticks)
#day_fiveteen(False, 2, False)

def day_sixteen(verbose, part):
    input = 19617804207202209144916044189917
    pattern= [0, 1, 0, -1]
    
    def FTT(input):
        output = ''
        signal = [int(s) for s in input]
        l_pattern = len(signal)
        if verbose: print("Input:", input)
        
        for i in range(l_pattern): # neues signal berechnen
            used_pattern = list()
            idx = 0 
            while True: # pattern für das signal
                p = [pattern[idx]] * (i+1)
                used_pattern += p  
                idx = idx+1 if idx+1 != len(pattern) else 0
                if len(used_pattern) > l_pattern:
                    break
            used_pattern = used_pattern[1:l_pattern+1]
            if verbose: print("Pattern:", used_pattern)
            
            num = sum([signal[i] * used_pattern[i] for i in range(l_pattern)])
            output += str(num)[-1]
        return output
    
    def run_phase(input, phase=100):
        for _ in range(phase):
            input = FTT(input)
        return input
    
    day_16_input = '59708372326282850478374632294363143285591907230244898069506559289353324363446827480040836943068215774680673708005813752468017892971245448103168634442773462686566173338029941559688604621181240586891859988614902179556407022792661948523370366667688937217081165148397649462617248164167011250975576380324668693910824497627133242485090976104918375531998433324622853428842410855024093891994449937031688743195134239353469076295752542683739823044981442437538627404276327027998857400463920633633578266795454389967583600019852126383407785643022367809199144154166725123539386550399024919155708875622641704428963905767166129198009532884347151391845112189952083025'
    if part == 1:    
        print("Test 1", run_phase(str(80871224585914546619083218645595))[:8])
        print("Test 2", run_phase(str(19617804207202209144916044189917))[:8])
        print("Test 3", run_phase(str(69317163492948606335995924319873))[:8])
        print()
        print("Answer", run_phase(day_16_input)[:8])
    elif part == 2:
        from itertools import cycle, accumulate
        
        def smart_FTT(input):
            digits = [int(i) for i in input]
            offset = int(input[:7])
            
            l = 10000 * len(input) - offset
            i = cycle(reversed(digits))
            arr = [next(i) for _ in range(l)]
            for _ in range(100):
                arr = [n % 10 for n in accumulate(arr)]
            return ''.join(str(i) for i in arr[-1:-9:-1])
        # wieso geht das ohne Phasen Berechnung????
        
        print("Test 1:", smart_FTT("03036732577212944063491565474664"))
        print("Test 2:", smart_FTT("02935109699940807407585447034323"))
        print("Test 3:", smart_FTT("03081770884921959731165446850517"))
        print()
        print("Answer:", smart_FTT(day_16_input))
#day_sixteen(False, 2)

def day_seventeen(verbose, part):
    data = read_input('17_input.txt')
    opcode = [int(d) for d in data[0].split(',')]
    
    def update_camera(input):
        camera = dict()
        x, y, width, height = 0, 0, 0, 0
        for code in input:
            if code == '\n': 
                print()
                y += 1
                width = max(width, x)
                x = 0
            else: 
                print(code, end='')
                camera[(x,y)] = code
                x += 1
        height = y-1
        return camera, x, y, width, height

    if part == 1:
        vm = Intcode(opcode)
        machine = vm.run()
        while True:
            try:
                next(machine)
            except StopIteration:
                break
        output = vm.pop_out()
        
        print("ASCII")
        ascii = [chr(code) for code in output]
        camera, x, y, width, height = update_camera(ascii)
        scaffold = []
        for i_y in range(1,height-1):
            for i_x in range(1,width-1):
                if camera[(i_x,i_y)]   == '#' and \
                   camera[(i_x-1,i_y)] == '#' and \
                   camera[(i_x+1,i_y)] == '#' and \
                   camera[(i_x,i_y-1)] == '#' and \
                   camera[(i_x,i_y+1)] == '#':
                        scaffold.append((i_x,i_y))
        
        answer = sum([x*y for (x,y) in scaffold])
        print("Sum of alignment parameters:", answer)
        
    elif part == 2:
        def translate_ascii(txt):
            ret = []
            for t in txt:
                ret.append(ord(t))
                ret.append(ord(','))
            ret[-1] = 10 # new line
            return ret
        
        def compress(string):
            a, b, c = list(), list(), list()
            ret = string
            return ret
        
        opcode[0] = 2
        camera = dict()
        x, y = 0, 0
        
        total_movement = "R,8,R,10,R,10,R,4,R,8,R,10,R,12,R,8,R,10,R,10,R,12,R,4,L,12,L,12,R,8,R,10,R,10,R,4,R,8,R,10,R,12,R,12,R,4,L,12,L,12,R,8,R,10,R,10,R,4,R,8,R,10,R,12,R,12,R,4,L,12,L,12"
        
        print(compress(total_movement))
        
        main_movement = list("A,B,A,C,A,B,C,A,B,C\n")
        main_a = list("R,8,R,10,R,10\n")
        main_b = list("R,4,R,8,R,10,R,12\n")
        main_c = list("R,12,R,4,L,12,L,12\n")
        video = list("n\n")
        inputs = [main_movement, main_a, main_b, main_c, video]
        for i, l in enumerate(inputs):
            new_l = []
            for d in l:
                new_l.append(ord(d))
            inputs[i] = new_l
  
        i = 0
        vm = Intcode(opcode)
        machine = vm.run()
        while True:
            output = vm.pop_out()
            ascii = [chr(code) for code in output]
            #update_camera(ascii)
            if verbose: print(f"Input {i}: {inputs[i]}")
            vm.push_in(inputs[i])
            i += 1
            
            try:
                next(machine)
            except StopIteration:
                break

        output = vm.pop_out()
        ascii = [chr(code) for code in output]
        #update_camera(ascii)  
        print("Star dust:", output[-1])
#day_seventeen(False, 1)

def day_eightteen(verbose, part):
    from collections import deque 
    
    map_raw = """#########
#b.A.@.a#
#########"""
    map = dict()
    doors = dict()
    keys = dict()
    position = (0,0)
    x, y = 0, 0
    for s in map_raw:
        if s == '\n':
            x = 0
            y += 1
            continue
        map[(x,y)] = s
        if s.isupper(): # door
            doors[s] = (x,y)
        elif s.islower():
            keys[s] = (x,y)
        elif s == '@':
            position = (x,y)
        x += 1
        
    def print_map():
        for k, v in map.items():
            x, y = k
            if x == 0 and y != 0: print()
            print(v, end='')
        print()   
    
    if verbose: print_map()
    
    # BFS
    look_at = deque([position])
    visited = set()
    while look_at:
        current = look_at.popleft()
        if current in visited:
            continue
        visited.add(current)
        
        #if map[current] == 
        
        x, y = current
#day_eightteen(True, 1)

def day_nineteen(verbose, part):
    data = read_input('19_input.txt')
    opcode = [int(d) for d in data[0].split(',')]
    
    def run_code(x, y):
        vm = Intcode(opcode.copy(), [x,y])
        try:
            next(vm.run())
        except StopIteration:
            pass
        output = vm.pop_out()
        return output[-1]
        
    if part == 1:
        affected = 0
        for y in range(50):
            for x in range(50):
                output = run_code(x, y)
                if output == 1: 
                    affected += 1
                    print(x,y)
        print("Points affected:", affected)
    elif part == 2:
        # square is 100x100
        # 2x2: x=16 y=9
        points = dict()
        size = 100
        x, y = 0, 0
        found_x = -1
        found_ship = False
        while True:
            output = run_code(x,y)
            points[(x,y)] = output
            if output == 1:
                found_x = x
                x_l, y_o = x-size+1, y-size+1
                output += run_code(x_l, y)   # unten links
                output += run_code(x, y_o)   # oben rechts
                output += run_code(x_l, y_o) # oben links
                if output == 4:
                    points[(x,y)] = 2
                    points[(x_l, y)] = 2
                    points[(x, y_o)] = 2
                    points[(x_l, y_o)] = 2
                    found_ship = True
                    if verbose:
                        for marky in range(y_o,y+1):
                            for markx in range(x_l,x+1):
                                points[(markx,marky)] = 2
            elif output == 0 and found_x > -1:
                x = max(0,found_x-(size*2))
                y += 1
                found_x = -1
                continue 
            elif output == 0 and found_x == -1 and x > 10 and y < 5:
                x = 0
                y += 1
                continue            
            if found_ship: break
            x += 1
            
        if verbose:
            for yr in range(y+1):
                print(str(yr).zfill(3) + ':  ', end='')
                for xr in range(x+size):
                    if not (xr,yr) in points: 
                        print('.', end='')
                        continue
                    out = points[(xr,yr)]
                    if out == 1: print('#', end='')
                    elif out == 0: print('.', end='')
                    elif out == 2: print('O', end='')
                    else: print('?', end='')
                print()
        print("Coordinates:", (x_l,y_o)) 
        print("Required sum:", x_l*10000+y_o)
#day_nineteen(False, 2)

def day_twenty(verbose, part):
    from collections import deque
    
    class SquareGrid:
        def __init__(self, width, height):
            self.width = width
            self.height = height
            self.passages = []
            self.weights = {}
            
        def in_bounds(self, id):
            (x, y) = id
            return 0 <= x < self.width and 0 <= y < self.height
            
        def passable(self, id):
            return id in self.passages
            
        def neighbors(self, id):
            (x, y) = id
            results = [(x+1, y), (x, y+1), (x-1, y), (x, y-1)]
            results = filter(self.in_bounds, results)
            results = filter(self.passable, results)
            return results
            
        def cost(self, to_node):
            return self.weights.get(to_node, 1)
    
    maze = """         A           
         A           
  #######.#########  
  #######.........#  
  #######.#######.#  
  #######.#######.#  
  #######.#######.#  
  #####  B    ###.#  
BC...##  C    ###.#  
  ##.##       ###.#  
  ##...DE  F  ###.#  
  #####    G  ###.#  
  #########.#####.#  
DE..#######...###.#  
  #.#########.###.#  
FG..#########.....#  
  ###########.#####  
             Z       
             Z      """
    
    def parse_maze(data):
        map = dict()
        portals = dict()
        raw_map = dict()
        for y, line in enumerate(data.split('\n')):
            for x, d in enumerate(line):
                raw_map[(x,y)] = d
        
        for (x,y), val in raw_map.items():
            if val == '.':
                map[(x,y)] = val
            elif val.isupper():
                y0 = raw_map[(x,y)]
                y1 = raw_map[(x,y-1)] if (x,y-1) in raw_map else ' '
                y2 = raw_map[(x,y+1)] if (x,y+1) in raw_map else ' '
                x0 = raw_map[(x,y)]
                x1 = raw_map[(x-1,y)] if (x-1,y) in raw_map else ' '
                x2 = raw_map[(x+1,y)] if (x+1,y) in raw_map else ' '
                
                if (y1 in (' ','#') or y2 in (' ','#')) and (x1 in (' ','#') or x2 in (' ','#')):
                    continue
                elif y1 in (' ','#') or y2 in (' ','#'): 
                    if x1.isupper(): 
                        p = x1+x0
                    else: 
                        p = x0+x2
                elif x1 in (' ','#') or x2 in (' ','#'): 
                    if y1.isupper(): 
                        p = y1+y0
                    else: 
                        p = y0+y2
                    
                if p in portals:
                    portals[p].append((x,y))
                else:
                    portals[p] = [(x,y)]
               
                map[(x,y)] = p

                
        return map, portals                
            
    map, portals = parse_maze(maze)

    start = portals['AA'][0]
    to_visit = deque([[start]])
    while to_visit:
        path = to_visit.popleft()
        #print(path)
        current = path[-1]
        x,y = current
        
        for i in range(4):
            if i == 0: point = (x+1,y)
            elif i == 1: point = (x-1,y)
            elif i == 2: point = (x,y+1)
            elif i == 3: point = (x,y-1)
            
            if point in map and point not in path:
                val = map[point]
                if val == 'ZZ':
                    print("Found goal!")
                    print(path)
                    print(len(path))
                    foundgoal = True
                    break
                elif val.isupper(): # portal
                    next = portals[val]
                    portal = [n for n in next if n != point]
                    to_visit.append(path + portal)
                else:
                    to_visit.append(path + [point])
day_twenty(True, 1)

def day_twentyone(verbose, part):
    data = read_input('21_input.txt')
    opcode = [int(d) for d in data[0].split(',')]
    
    def translate_ascii(string):
        out = []
        for s in string:
            out.append(ord(s))
        out += [10]
        return out
    
    vm = Intcode(opcode)
    machine = vm.run()

    program = ""
    if part == 1:
        program = """NOT C T
AND D T
OR T J
NOT A T
OR T J
WALK"""
    
    elif part == 2:
        program = """NOT A T
OR T J
NOT C T
AND D T
AND H T
OR T J
NOT B T
AND C T
AND D T
OR T J
RUN"""
    
    vm.push_in(translate_ascii(program))
    while True:
        output = vm.pop_out()
        ascii = [chr(code) for code in output]
        
        try:
            next(machine)
        except StopIteration:
            break
    output = vm.pop_out()
    string = ''
    for s in output:
        if s >= 128: continue
        string += chr(s)
    print(string, end='')
    print("Hull damage:", output[-1])
#day_twentyone(True, 2)

def day_twentytwo(verbose, part):
    if part == 1:
        from collections import deque
        
        def deal_new_stack(cards):
            if verbose: print("deal into new stack")
            new_cards = cards.copy()
            new_cards.reverse()
            return new_cards
            
        def cut_n_cards(cards, N):
            if verbose: print("cut", N)
            top_n = cards[:N]
            bottom_n = cards[N:]
            return bottom_n + top_n
        
        def deal_with_increment(cards, N):
            if verbose: print("deal with increment", N)
            new_cards = [-1] * len(cards)
            ccards = deque(cards.copy())
            pos = 0
            for _ in range(len(cards)):
                new_cards[pos] = ccards.popleft()
                pos += N
                if pos >= len(cards): pos -= len(cards)
            return new_cards
        
        val = list(range(10007))
        shuffle = read_input('22_input.txt')
        position = 2019
        
    elif part == 2:
        print("Nope...")
        return
        
    for line in shuffle:
        if "cut" in line:
            n = int(line.split(' ')[1])
            val = cut_n_cards(val, n)
        elif "deal into new stack" in line:
            val = deal_new_stack(val)
        elif "deal with increment" in line:
            n = int(line.split(' ')[-1])
            val = deal_with_increment(val, n)
    if verbose: print("Result:", val)
    
    if part == 1:
        for pos, card in enumerate(val):
            if card == position: break
    elif part == 2: 
        pos = val
    
    print("Card is at position", pos)
#day_twentytwo(True, 2)

def day_twentythree(verbose, part):
    pass
day_twentythree(True, 1)

def day_twentyfour(verbose, part):
    pass
day_twentyfour(True, 1)

def day_twentyfive(verbose, part):
    pass
day_twentyfive(True, 1)