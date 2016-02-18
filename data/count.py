import glob

def count(t):
    fin = open('./graph/' + str(t) + '.csv')
    p_map = {}
    lines = fin.readlines()
    for l in lines:
        ls = l.split(',')
        x = int(ls[0]); y = int(ls[1])
        p_map[x] = 0; p_map[y] = 0
    fin.close()
    print str(t) + '\t' + str(len(p_map))

if __name__ == '__main__':
    for i in range(120):
        count(i)

