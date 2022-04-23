import re
import functools

def dates(lines):
    out = []
    for file in lines:
        m = re.findall(r"([0-9]{6}_[0-9]{6}).wav", file)
        if len(m) == 1:
            date = m[0]
            out.append(date)
        else:
            raise Exception(file, m)
    return out

def compare(a, b):
    def date(s):
        m = re.findall(r"([0-9]{6}_[0-9]{6}).wav", s)
        if len(m) == 1:
            date = m[0]
            return date
        else:
            raise Exception(file, m)

    if date(a) == date(b):
        if a > b:
            return 1
        elif a < b:
            return -1
        else:
            return 0
    elif date(a) > date(b):
        return 1
    else:
        return -1

with open("./filenames.txt", "r") as file:
    contents = file.read()
    lines = contents.split("\n")
    sorted_lines = sorted(lines, key=functools.cmp_to_key(compare))
    with open("output.txt", "w") as file:
        file.write("\n".join(sorted_lines))
        
    
    # ds = dates(lines)

    # import numpy as np

    # sortedargs = np.argsort(ds, axis=0)
    # s = np.array(lines)[sortedargs]
    # print(s)
        
