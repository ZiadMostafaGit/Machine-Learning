def preprocessing(data, option = 1):
    '''
    Args:
        data: numpy array examples x features
        option: 1 for MinMaxScaler and 2 for StandardScaler

    Returns: preprocessed data
    '''
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    if option == 1:
        processor = MinMaxScaler()
    elif option == 2:
        processor = StandardScaler()
    else:
        return data, None # don't process

    return processor.fit_transform(data), processor


def load_data(data_path, preprocessing_option = 1):
    import pandas as pd
    df = pd.read_csv(data_path)
    data = df.to_numpy()

    x = data[:, :3]
    # in next projects we will better reshape to (M, 1)
    # as this one makes broadcast errors
    t = data[:, -1]

    x, _ = preprocessing(x, preprocessing_option)

    return df, data, x, t

