import sys

def cut():
    fin = open('user_info.dat')
    fout = open('user_info.dat2', 'w')
    lines = fin.readlines()
    for l in lines:
        ls = l.split('\t')
        newline = ls[1]
        for i in range(2, len(ls)):
            newline = newline + '\t' + ls[i]
        fout.write(newline)
    fin.close()
    fout.close()

if __name__ == '__main__':
    cut()

