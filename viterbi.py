import numpy as np

def run_viterbi(emission_scores, trans_scores, start_scores, end_scores):
    """Run the Viterbi algorithm.

    N - number of tokens (length of sentence)
    L - number of labels

    As an input, you are given:
    - Emission scores, as an NxL array
    - Transition scores (Yp -> Yc), as an LxL array
    - Start transition scores (S -> Y), as an Lx1 array
    - End transition scores (Y -> E), as an Lx1 array

    You have to return a tuple (s,y), where:
    - s is the score of the best sequence
    - y is the size N array/seq of integers representing the best sequence.
    """
    L = start_scores.shape[0]
    assert end_scores.shape[0] == L
    assert trans_scores.shape[0] == L
    assert trans_scores.shape[1] == L
    assert emission_scores.shape[1] == L
    N = emission_scores.shape[0]

    score = np.zeros((N,L))

    path = np.zeros((N,L))
    y = []

    for i in range(L):
        score[0][i] = start_scores[i] + emission_scores[0][i]

    #R(i, yi) = maxOverYi-1(e(xi|yi) * t(yi|yi - 1) * R(i - 1,yi - 1))
    #O(NL^2) algorithmx
    for i in range(1, N):
        for j in range(L):
            max_score = -np.inf
            for k in range(L):
                temp = score[i - 1][k] + emission_scores[i][j] + trans_scores[k][j]
                if(temp > max_score):
                    score[i][j] = temp
                    path[i - 1][j] = k
                    max_score = temp

    max_end = -np.inf
    last_ind = -1
    retScore = 0
    for i in range(L):
        temp = score[N-1][i] + end_scores[i]
        if(temp > max_end):
            last_ind = i
            max_end = temp
    y.append(last_ind)

    if(N > 1):
        prev_ind = last_ind
        for i in range(N - 2, -1, -1):
            m = int(path[i][prev_ind])
            y.append(m)
            prev_ind = m

    y.reverse()
    # score set to 0
    return (max_end, y)

