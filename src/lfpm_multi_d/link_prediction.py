import random
import math

K = 15
T = 17
max_iter = 99
seed = 2    # cross validation
random.seed(2)
res = {}

def inner_product(arr1, arr2):
    res = 0.0
    for k in range(K):
        res += arr1[k] * arr2[k]
    return res

def lp(t):
    # read h/hp
    h = {}; hp = {}
    fin = open('./res/h_' + str(t) + '_' + str(max_iter) + '.txt')
    lines = fin.readlines()
    for line in lines:
        ls = line.split(' ')
        x = int(ls[0]); v = []
        for k in range(K):
            v.append(float(ls[1+k]))
        h[x] = v
    fin.close()
    fin = open('./res/h_p_' + str(t) + '_' + str(max_iter) + '.txt')
    lines = fin.readlines()
    for line in lines:
        ls = line.split(' ')
        x = int(ls[0]); v = []
        for k in range(K):
            v.append(float(ls[1+k]))
        hp[x] = v
    fin.close()

    # softmax
    den = {}
    for x in h:
        den[x] = 0.0
        for y in hp:
            ins_exp = inner_product(h[x], hp[y])
            den[x] += math.exp(ins_exp)
    probs = {}
    for x in h:
        probs[x] = {}
        for y in hp:
            ins_exp = inner_product(h[x], hp[y])
            probs[x][y] = math.exp(ins_exp) / den[x]

    # read (positive) links
    pos_links = {}; neg_links = {}
    for x in h:
        pos_links[x] = []
        neg_links[x] = []
    fin = open('../../data_sm/nips_17/out/' + str(seed) + '/' + str(t) + '.test.csv')
    lines = fin.readlines()
    for line in lines:
        ls = line.split(',')
        x = int(ls[0])
        y = int(ls[1])
        if x in pos_links:
            pos_links[x].append(y)
        if y in pos_links:
            pos_links[y].append(x)
    fin.close()

    all_set = set([i for i in pos_links])
#    for x in pos_links:
    for x in h:
        pos_set = set(pos_links[x])
        if len(pos_set) == 0:
            continue
        neg_set = all_set - pos_set
        neg_set.remove(x)
        twenty_test = random.sample(list(neg_set), int(0.2*len(neg_set)))
        '''
        if len(neg_set) > len(pos_set):     # use equal number of pos/neg
            l = len(neg_set)
            while l > len(pos_set):
                neg_set.remove(random.choice(list(neg_set)))
                l = len(neg_set)
        '''
#        neg_links[x] = list(neg_set)
        neg_links[x] = twenty_test

    # compute AUC
    for x in pos_links:
        pos_set = pos_links[x]; neg_set = neg_links[x]
        for y in pos_set:
            prob = probs[x][y]
            res[prob] = 1
        for y in neg_set:
            prob = probs[x][y]
            res[prob] = -1


def evaluate():
    new_x = 0.0; new_y = 0.0; old_x = 0.0; old_y = 0.0; auc = 0.0
    pos_num = len([k for (k,v) in res.items() if v > 0])
    neg_num = len([k for (k,v) in res.items() if v < 0])

    for e in sorted(res.items(), key = lambda x: x[0], reverse = True):
        if e[1] > 0:
            new_y += 1.0/pos_num
        else:
            new_x += 1.0/neg_num
        auc += new_y * (new_x - old_x)
        old_x = new_x
        old_y = new_y
#        print e[0], e[1]
    print auc, new_y, new_x, pos_num, neg_num, len(res)


if __name__ == '__main__':
#    lp(14)
    for t in range(T):
        lp(t)
    evaluate()

