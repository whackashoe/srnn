import numpy as np

#slice sequences into many subsequences
def seqs_split(seqs):
    ret = []

    for i in range(seqs.shape[0]):
        split1 = np.split(seqs[i], 8)
        a = []

        for j in range(8):
            s = np.split(split1[j],8)
            a.append(s)

        ret.append(a)

    return ret

