



if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    # Create a dataset
    np.random.seed(0)  # For reproducibility
    time = np.arange(0, 100, 1)  # Time dimension (x)
    noise = np.random.normal(0, 1, size=time.shape)  # Random noise
    signal = np.sin(time * 0.1) + noise  # Non-smooth signal (y)

    # Exponentially Weighted Averages
    def ewa(data, alpha=0.9):
        smoothed_data = np.zeros(data.shape)
        smoothed_data[0] = data[0]
        for i in range(1, len(data)):
            smoothed_data[i] = alpha * smoothed_data[i - 1] + (1 - alpha) * data[i]
        return smoothed_data

    alpha = 0.9
    smoothed_signal = ewa(signal, alpha)

    import pandas as pd
    signal_series = pd.Series(signal)


    #smoothed_signal = signal_series.ewm(alpha=alpha, adjust=False).mean()
    #smoothed_signal = signal_series.ewm(span=10, adjust=False).mean()


    # Plotting
    plt.figure(figsize=(14, 6))
    plt.plot(time, signal, label='Original Signal')
    plt.plot(time, smoothed_signal, label='Smoothed Signal with EWA', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.title('Signal Smoothing using Exponentially Weighted Averages')
    plt.legend()
    plt.show()