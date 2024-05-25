from sklearn.metrics import average_precision_score


if __name__ == '__main__':

    y_prop = [0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91] # sorted
    print(average_precision_score([1, 0, 0, 0, 0, 0, 0, 0, 0], y_prop))  # 1
    print(average_precision_score([1, 1, 1, 1, 0, 0, 0, 0, 0], y_prop))  # 1
    print(average_precision_score([1, 0, 0, 0, 0, 0, 0, 0, 1], y_prop))  # 0.611
    print(average_precision_score([0, 1, 0, 0, 0, 0, 0, 0, 0], y_prop))  # 1/2
    print(average_precision_score([0, 1, 0, 1, 0, 0, 0, 0, 0], y_prop))  # 1/2
    print(average_precision_score([0, 1, 0, 1, 0, 1, 0, 1, 0], y_prop))  # 1/2 each 2nd
    print(average_precision_score([0, 0, 1, 0, 0, 0, 0, 0, 0], y_prop))  # 1/3
    print(average_precision_score([0, 0, 1, 0, 0, 1, 0, 0, 0], y_prop))  # 1/3
    print(average_precision_score([0, 0, 1, 0, 0, 1, 0, 0, 1], y_prop))  # 1/3 each 3rd elem
    print(average_precision_score([0, 0, 0, 1, 0, 0, 0, 1, 0], y_prop))  # 1/4 each 4th elem
    print(average_precision_score([0, 0, 0, 0, 0, 0, 0, 0, 1], y_prop))  # 0.11
    print(average_precision_score([0, 0, 0, 0, 0, 0, 0, 1, 1], y_prop))  # 0.173
    print(average_precision_score([1, 0, 0, 0, 0, 0, 0, 1, 1], y_prop))  # 0.527
    print(average_precision_score([1, 1, 0, 0, 0, 1, 0, 1, 1], y_prop))  # 0.71
    print(average_precision_score([1, 1, 0, 0, 0, 0, 1, 1, 1], y_prop))  # 0.696
    print(average_precision_score([1, 1, 1, 1, 1, 1, 1, 1, 0], y_prop))  # 1.00
    print(average_precision_score([1, 1, 1, 1, 1, 1, 1, 0, 1], y_prop))  # 0.98
    print(average_precision_score([1, 1, 1, 1, 1, 1, 0, 1, 1], y_prop))  # 0.97
    print(average_precision_score([1, 1, 1, 1, 1, 0, 1, 1, 1], y_prop))  # 0.95
    print(average_precision_score([1, 1, 1, 1, 0, 1, 1, 1, 1], y_prop))  # 0.93
    print(average_precision_score([1, 1, 1, 0, 1, 1, 1, 1, 1], y_prop))  # 0.90
    print(average_precision_score([1, 1, 0, 1, 1, 1, 1, 1, 1], y_prop))  # 0.87
    print(average_precision_score([1, 0, 1, 1, 1, 1, 1, 1, 1], y_prop))  # 0.83
    print(average_precision_score([0, 1, 1, 1, 1, 1, 1, 1, 1], y_prop))  # 0.77
