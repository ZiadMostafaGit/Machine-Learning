import numpy as np

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
    t = data[:, -1]

    x, _ = preprocessing(x, preprocessing_option)
    t, test_processor = preprocessing(t.reshape((-1, 1)), preprocessing_option)
    t = t.reshape(-1)

    return df, data, x, t, test_processor

