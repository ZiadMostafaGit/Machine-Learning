import argparse
import time
import numpy as np
import pandas as pd
from collections import Counter
import os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, precision_score, recall_score


def data_processor(root_dir):
    def process_pandas(df):
        df.columns = map(str.lower, df.columns)
        df.rename(columns={'class': 'label'}, inplace=True)

        df = df.drop('time', axis=1)  # logically not that important?
        df['amount'] = np.log1p(df.amount)  # others looks scaled

        X = df[df.columns[:-1].tolist()]
        y = df[df.columns[-1]]
        return X, y

    df_train = pd.read_csv(os.path.join(root_dir, 'split/train.csv'))
    df_test = pd.read_csv(os.path.join(root_dir, 'split/val.csv'))

    X_train, y_train = process_pandas(df_train)
    X_test, y_test = process_pandas(df_test)

    print(f"y_train {Counter(y_train)}")
    print(f"y_test  {Counter(y_test)}")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    np.random.seed(42)

    parser = argparse.ArgumentParser(description='Credit Fraud Evaluator')

    parser.add_argument('--root_dir', type=str,
                        default='/home/moustafa/0hdd/00Udemy/MachineLearning/0code/7-binary-classification/15-project-credit-card-fraud')

    args = parser.parse_args()
    X_train, X_test, y_train, y_test = data_processor(args.root_dir)
    X_train, X_test, y_train, y_test = X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

    if False:
        # we can reduce the dimensions.
        # I found 18 as good or close to no compression
        from sklearn.decomposition import PCA

        pca = PCA(n_components=18, random_state=42)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    # split data into positive and negative parts
    X_train_positive = X_train[y_train == 1]
    y_train_positive = y_train[y_train == 1]

    X_train_negative = X_train[y_train == 0]
    y_train_negative = y_train[y_train == 0]

    # we can't just use the whole data as it is in the inference
    # 2 ways to reduce
    # 1) Just sample some of the data
    # 2) Do k-means to gather them in clusters
    # surprisingly both works with similar performance!
    do_sampling = True

    if do_sampling:
        indices = np.random.choice(len(X_train_negative), 305, replace=False)
        X_train_negative = X_train_negative[indices]
        y_train_negative = y_train_negative[indices]
    else:
        kmeans_negative = KMeans(n_clusters=305)
        kmeans_negative.fit(X_train_negative)
        print('Finished fitting K means')

        X_train_negative = kmeans_negative.cluster_centers_
        y_train_negative = np.zeros(len(kmeans_negative.cluster_centers_))

    # merge the data again
    X_train = np.vstack((X_train_positive, X_train_negative))
    y_train = np.concatenate((y_train_positive, y_train_negative))
    print(f'Updated data stats: {Counter(y_train)}')

    knn = KNeighborsClassifier(n_neighbors=100).fit(X_train, y_train)

    start = time.time()
    y_pred = knn.predict(X_test)
    print(f'KNN stats: {Counter(y_pred)}')

    f1 = f1_score(y_test, y_pred)      # , average='weighted'
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f'KNN f1 score {f1 * 100:.2f}% - '
          f' - precision {precision * 100:.2f}%'
          f' - recall {recall * 100:.2f}%'
          )
    print(f"Inference Time is {(time.time() - start):.0f} seconds")
