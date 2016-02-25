# classification on users
import os.path
import sys

label = {}

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

def classification(start, end, jump):
    read_dict()
    for i in range(120):    # year
        predict = {}
        n_pos = 0.0; n_neg = 0.0
        for n_iter in range(start, end, jump):      # iteration
            filename = './save/' + str(n_iter) + '/' + str(i) + '.txt'
            if not os.path.isfile(filename):
                break
            fin = open(filename)
            lines = fin.readlines()
            n_pos = 0.0; n_neg = 0.0
            for line in lines:
                ls = line.split(' ')
                old_id = int(ls[0])
                if old_id not in label:
                    continue
#                k = float(ls[1]) + float(ls[2])
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
        if n_pos == 0:
            continue
        print i,
#        print n_pos, n_neg

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
#            print label[old_id],
#        gu = raw_input()
#        print newX, newY,
        if auc < 0.5:
            auc = 1-auc
        print auc,

        # classification 
        cor = 0.0; tot = 0.0
        for old_id in predict:
            if label[old_id] * predict[old_id] > 0:
                cor += 1
            tot += 1
        acc = cor/tot
        if acc < 0.5:
            acc = 1-acc
        print acc

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print 'Usage: python classification.py <start> <end> <jump>'
        print 'Example: python classification.py 80 100 2'
        sys.exit()
    start = int(sys.argv[1])
    end = int(sys.argv[2])
    jump = int(sys.argv[3])
    print '-------\nsorting may be a problem: lots of identical values\n-------'
    print 't\tauc \t classification'
    classification(start, end, jump)


