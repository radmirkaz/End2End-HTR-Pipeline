import numpy as np
import editdistance


def string_accuracy(y_true, y_pred):
    scores = []

    for true, pred in zip(y_true, y_pred):
        scores.append(true == pred)

    avg_score = np.mean(scores)

    return avg_score


def cer(y_true, y_pred):
    scores = []

    for p_seq2, p_seq1 in zip(y_true, y_pred):
        p_vocab = set(p_seq1 + p_seq2)

        p2c = dict(zip(p_vocab, range(len(p_vocab))))

        c_seq1 = [chr(p2c[p]) for p in p_seq1]
        c_seq2 = [chr(p2c[p]) for p in p_seq2]

        error = editdistance.eval(''.join(c_seq1), ''.join(c_seq2)) / len(c_seq2)

        scores.append(error)

    avg_score = np.mean(scores)

    return avg_score
