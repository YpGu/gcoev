# classification on users
# only on users who stay in congress for > 1 term

import os.path
import sys

label = {}
udict = {}  # new_id -> old_id

def read_dict():
    fin = open('../../data/dict/user_info.dat')
    lines = fin.readlines()
    for line in lines:
        ls = line.split('\t')
        old_id = int(ls[-1])
        party = ls[3].lower()
        if party == 'democrat':
            label[old_id] = -1
        else:
            label[old_id] = 1
    fin.close()

def read_id_transform():
    fin = open('../../data/dict/user_id_map.dat')
    lines = fin.readlines()
    for line in lines:
        ls = line.split(',')
        old_id = int(ls[1])
        new_id = int(ls[0])
        udict[new_id] = old_id
    fin.close()

# has prodecessor
def read_dict_year(year):
    label_y = {}
    filename = '../../data/dict/has_prodecessor/' + str(year) + '.txt'
    if not os.path.isfile(filename):
        return {}
    fin = open(filename)
    lines = fin.readlines()
    for line in lines:
        new_id = int(line)
        old_id = udict[new_id]
        if old_id in label:
            label_y[old_id] = label[old_id]
    fin.close()

    return label_y

def classification(file_prefix):
    read_dict()
    read_id_transform()

    for i in range(120):    # year
        label_y = read_dict_year(i)
        predict = {}
        n_pos = 0.0; n_neg = 0.0
#        filename = './save/baseline_0/' + str(i) + '.txt'
        filename = file_prefix + str(i) + '.txt'
        if not os.path.isfile(filename):
            continue
        fin = open(filename)
        lines = fin.readlines()
        n_pos = 0.0; n_neg = 0.0
        for line in lines:
            ls = line.split(' ')
            old_id = int(ls[0])
#            if old_id not in label:
            if old_id not in label_y:
                continue
            k = float(ls[1])
            if old_id in predict:
                predict[old_id] += k
            else:
                predict[old_id] = k

            if label[old_id] == 1:
                n_pos += 1
            elif label[old_id] == -1:
                n_neg += 1
        fin.close()
        print i,

        # auc
        newX = 0.0; newY = 0.0; oldX = 0.0; oldY = 0.0; auc = 0.0
#        for dat in sorted(predict.items(), key = lambda x: 100 * x[1] + label[x[0]], reverse = True):
        for dat in sorted(predict.items(), key = lambda x: x[1], reverse = True):
            old_id = dat[0]
            if label[old_id] == 1:
                newY += 1/n_pos
            else:
                newX += 1/n_neg
            auc += newY * (newX - oldX)
            oldX = newX; oldY = newY
        if auc < 0.5:
            auc = 1-auc
        print auc,

        # classification 
        cor = 0.0; tot = 0.0
        for old_id in predict:
            if label[old_id] * predict[old_id] > 0:
                cor += 1
            tot += 1
        if tot != 0:
            acc = cor/tot
            if acc < 0.5:
                acc = 1-acc
            print acc
        else:
            print ''

if __name__ == '__main__':
    '''
    if len(sys.argv) != 4:
        print 'Usage: python classification.py <start> <end> <jump>'
        print 'Example: python classification.py 80 100 2'
        sys.exit()
    start = int(sys.argv[1])
    end = int(sys.argv[2])
    jump = int(sys.argv[3])
    print '-------\nsorting may be a problem: lots of identical values\n-------'
    '''
#    print 't\tauc \t classification'
    classification(sys.argv[1])


