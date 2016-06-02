# generate random negative links 

import random
import os
import math
import sys

random.seed(0)
SS = 1

T = 20
N = 1673
file_prefix = './dblp_venue/'

#T = 17
#N = 110
#file_prefix = './nips_17/out/'

#T = 33
#N = 200
#file_prefix = './covote/out/'

#T = 50
#N = 82
#file_prefix = './infocom/out/'

def gen_neg(t):
    print t
    # read (positive) links
    pos_links = {}; neg_links = {}

    fin = open(file_prefix + str(t) + '.csv')
    lines = fin.readlines()
    for line in lines:
        ls = line.split(',')
        x = int(ls[0])
        y = int(ls[1])
        if x in pos_links:
            pos_links[x].append(y)
        else:
            pos_links[x] = [y]
        if y in pos_links:
            pos_links[y].append(x)
        else:
            pos_links[y] = [x]
    fin.close()
    appeared = set([i for i in pos_links])

    all_set = set([i for i in range(N)])
    for x in pos_links:
        pos_set = set(pos_links[x])
        if len(pos_set) == 0:
            continue

#        neg_set = all_set - pos_set
        neg_set = appeared - pos_set
        neg_set.remove(x)
        neg_samples = list(neg_set)

        neg_links[x] = neg_samples
#        '''
        # comment if use all non-existing links
        # uncomment if use the same number of no-links as yes-links
        if len(neg_set) > len(pos_set) * SS:
            neg_samples = random.sample(list(neg_set), len(pos_set) * SS)
            #neg_samples = random.sample(list(neg_set), int(len(neg_set) * 0.2))
            neg_links[x] = neg_samples
        else:
            neg_links[x] = list(neg_set)
#        '''

    fout = open(file_prefix + str(t) + '.neg.csv', 'w')
    for x in neg_links:
        ns = neg_links[x]
        for y in ns:
            newline = str(x) + ',' + str(y) + '\n'
            fout.write(newline)
    fout.close()


if __name__ == '__main__':
    for t in range(T):
        gen_neg(t)

