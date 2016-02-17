import glob

valid_votes = ['yea', 'nay', 'aye', 'no']
vote_score = {'yea':1, 'nay':-1, 'aye':1, 'no':-1}
p_map = {}      # uid -> new_id
y_pos_neg = {}  # year -> [cov ... ]

def read_data():
    files = glob.glob('../../15D/voting_data/all/*')
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
    fout = open('./p_dict', 'w')
    for i in p_map:
        p_map[i] = new_id
        newline = str(new_id) + ',' + str(i) + '\n'
        fout.write(newline)
        new_id += 1
    fout.close()

 
def generate_graph():
    files = glob.glob('../../15D/voting_data/all/*')
    print len(files)
    fid = 0
    for filename in files:
        if fid % 1000 == 0:
            print fid
        fid += 1
#        print filename
        year = int(filename.split('/')[-1].split('_')[0]) - 1      # term (start from 0)
        if year not in y_pos_neg:
            y_pos_neg[year] = []
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
        for i in pos:
            for j in pos:
                if i < j:
                    cov[(i,j)] = 1
        for i in neg:
            for j in neg:
                if i < j:
                    cov[(i,j)] = 1
        y_pos_neg[year].append(cov)
    
    for y in y_pos_neg:
        # each year
        print y
        y_cov = {}
        covs = y_pos_neg[y]
        for cov in covs:
            # cov is a dictionary: (i,j) -> e
            for (i,j) in cov:
                if (i,j) in y_cov:
                    y_cov[(i,j)] += cov[(i,j)]
                else:
                    y_cov[(i,j)] = cov[(i,j)]
        fout = open('./graph/' + str(y) + '.csv', 'w')
        for (i,j) in y_cov:
            newline = str(i) + ',' + str(j) + ',' + str(y_cov[(i,j)]) + '\n'
            fout.write(newline)
        fout.close()

if __name__ == '__main__':
    read_data()
    generate_graph()

