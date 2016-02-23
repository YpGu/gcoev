import glob

valid_votes = ['yea', 'nay', 'aye', 'no']
vote_score = {'yea':1, 'nay':-1, 'aye':1, 'no':-1}
p_map = {}      # uid -> new_id
y_pos_neg = {}  # year -> [cov ... ]

def read_data():
    files = glob.glob('../../15D/voting_data/all/*')
    print len(files)
    fid = 0
    for filename in files:
        if fid % 1000 == 0:
            print fid
        fin = open(filename)
        lines = fin.readlines()
        for line in lines:
            ls = line.split('\t')
            uid = int(ls[0])
            res = ls[1].split('\n')[0].lower()
            if res not in vote_score:
                continue
            p_map[uid] = 0
        fin.close()
        fid += 1

    new_id = 0
    fout = open('./user_id_map.dat', 'w')
    for i in p_map:
        p_map[i] = new_id
        newline = str(new_id) + ',' + str(i) + '\n'
        fout.write(newline)
        new_id += 1
    fout.close()

 
def generate_graph(year):
    y_pos_neg = []
    files = glob.glob('../../15D/voting_data/all/' + str(year+1) + '_*')
    threshold = int(len(files) * 0.2)
    print len(files)

    for filename in files:
        year = int(filename.split('/')[-1].split('_')[0]) - 1      # term (start from 0)
        # each bill
        pos = []; neg = []
        fin = open(filename)
        lines = fin.readlines()
        for line in lines:
            ls = line.split('\t')
            res = ls[1].split('\n')[0].lower()
            if res not in vote_score:
                continue
            uid = p_map[int(ls[0])]
            score = vote_score[res]
            if score == 1:
                pos.append(uid)
            else:
                neg.append(uid)
        fin.close()
        # co-voting relationship for a bill (note: undirected)
        cov = {};
        # both vote +1
        for i in pos:
            for j in pos:
                if i < j:
                    cov[(i,j)] = 1
        # both vote -1
        for i in neg:
            for j in neg:
                if i < j:
                    cov[(i,j)] = 1
        # vote differently: 1 & -1
        for i in pos:
            for j in neg:
                if i < j:
                    cov[(i,j)] = -1
        y_pos_neg.append(cov)
    

    y_cov = {}
    for cov in y_pos_neg:
        # cov is a dictionary: (i,j) -> +1/-1  (agree/disagree)
        for (i,j) in cov:
            if (i,j) in y_cov:
                y_cov[(i,j)] += cov[(i,j)]
            else:
                y_cov[(i,j)] = cov[(i,j)]
    fout = open('./graph/' + str(year) + '.csv', 'w')
    for (i,j) in y_cov:
        if y_cov[(i,j)] <= threshold:
            continue            # they have not co-voted enough number of bills
#        y_cov[(i,j)] -= threshold
        newline = str(i) + ',' + str(j) + ',' + str(y_cov[(i,j)]) + '\n'
        fout.write(newline)
    fout.close()

if __name__ == '__main__':
    read_data()
    for i in range(0,120):
        print i,
        generate_graph(i)

