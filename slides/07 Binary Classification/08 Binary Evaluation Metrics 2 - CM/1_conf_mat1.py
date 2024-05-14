from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def cm():
    y_true = [0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1]     # ground
    y_pred = [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]     # model

    conf = confusion_matrix(y_true, y_pred)
    print(conf)

    tn, fp, fn, tp = conf.ravel()   # table order

    print(f'tp={tp}, fn={fn}, tn={tn}, fp={fp}')

    '''

    [[2 4]
     [5 3]]
    tn=2, fp=4, fn=5, tp=3

    '''

    #report = classification_report(y_true, y_pred)
    #print(report)


def compute_metrics(tp, tn, fp, fn):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fpr = fp / (fp + tn)
    acc = (tp + tn) / (tp + tn + fp + fn)

    print(f'tp={tp}, fn={fn}   |    tn={tn}, fp={fp}')
    print(f'\t\t\t\t\t\t\t\t\tprecision = {precision:.2f}   recall = {recall:.2f}   fpr = {fpr:.2f}   acc = {acc:.2f}')


if __name__ == '__main__1':
    cm()

if __name__ == '__main__':
    p, n = 200, 150     # total positive and negative gt

    tp, tn = 160, 100   # correct predictions
    compute_metrics(tp, tn, fp=n - tn, fn=p - tp)

    tp, tn = 160, 120
    compute_metrics(tp, tn, fp=n - tn, fn=p - tp)

    tp, tn = 160, 140
    compute_metrics(tp, tn, fp=n - tn, fn=p - tp)

    tp, tn = 160, 150
    compute_metrics(tp, tn, fp=n - tn, fn=p - tp)

    tp, tn = 120, 100   # correct predictions
    compute_metrics(tp, tn, fp=n - tn, fn=p - tp)

    tp, tn = 140, 100   # correct predictions
    compute_metrics(tp, tn, fp=n - tn, fn=p - tp)

    tp, tn = 160, 100   # correct predictions
    compute_metrics(tp, tn, fp=n - tn, fn=p - tp)

    tp, tn = 200, 100   # correct predictions
    compute_metrics(tp, tn, fp=n - tn, fn=p - tp)

    tp, tn = 200, 150   # correct predictions
    compute_metrics(tp, tn, fp=n - tn, fn=p - tp)

    tp, tn = 1, n
    compute_metrics(tp, tn, fp=n - tn, fn=p - tp)

    tp, tn = p, 0
    compute_metrics(tp, tn, fp=n - tn, fn=p - tp)
