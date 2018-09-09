import numpy as np

#slice sequences into many subsequences
def seqs_split(seqs, slice_width):
    ret = []

    for i in range(seqs.shape[0]):
        split1 = np.split(seqs[i], slice_width)
        a = []

        for j in range(slice_width):
            s = np.split(split1[j], slice_width)
            a.append(s)

        ret.append(a)

    return ret

