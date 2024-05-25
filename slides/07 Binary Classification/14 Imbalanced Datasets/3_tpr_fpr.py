

def compute_metrics(p, n, tp, fp):
    tn = n - fp
    fn = p - tp

    precision = tp / (tp + fp)
    recall = tp / (tp + fn) # tpr
    fpr = fp / n
    #acc = (tp + tn) / (tp + tn + fp + fn)

    print(f'tp={tp}, fn={fn}   |    tn={tn}, fp={fp}')
    print(f'\t\t\t\t\t\t\t\t\tprecision = {precision:.5f}   recall = {recall:.3f}   fpr = {fpr:.7f}')


if __name__ == '__main__':
    p, n = 1000, 1000000     # total positive and negative gt

    tp, fp = 75, 2000
    compute_metrics(p, n, tp, fp)

    tp, fp = 75, 100
    compute_metrics(p, n, tp, fp)   # hard to notice FPR improvements

    tp, fp = 150, 100
    compute_metrics(p, n, tp, fp)

    tp, fp = 300, 100
    compute_metrics(p, n, tp, fp)   # TPR jumps quickly

