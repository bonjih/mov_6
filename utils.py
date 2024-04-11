import os
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import cv2


def make_csv(data, path):
    if data:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        file_empty = not os.path.exists(path) or os.stat(path).st_size == 0

        data_out = pd.DataFrame({'time_roi_1': [data[0][0]], 'sum_features_roi_1': [data[0][1]],
                                 'result_roi_1': [data[0][2]], 'win_roi_1': [data[0][3]], 'psd_val_1': [data[0][4]],
                                 'time_roi_2': [data[1][0]], 'sum_features_roi_2': [data[1][1]],
                                 'result_roi_2': [data[1][2]], 'win_roi_2': [data[1][3]], 'psd_val_2': [data[1][4]]
                                 })
        if file_empty:
            data_out.to_csv(path, mode='w', index=False)
        else:
            data_out.to_csv(path, mode='a', header=False, index=False)


def make_plot():
    var1 = 'win_roi_1'
    var2 = 'win_roi_2'

    plt.figure(figsize=(10, 6))

    data = pd.read_csv('./output/audit_data.csv')

    # Mapping values to desired representation
    data['result_roi_1'] = data['result_roi_1'].map({-1: False, 1: True, 0: 'POT'})
    data['result_roi_2'] = data['result_roi_2'].map({-1: False, 1: True, 0: 'POT'})

    plt.plot(data['time_roi_1'], data[var1], color='blue', label=var1)
    plt.plot(data['time_roi_1'], data[var2], color='green', label=var2)

    for result, color in zip([False, True, 'POT'], ['yellow', 'red', 'grey']):
        if result != 'POT':
            plt.scatter(data[data['result_roi_1'] == result]['time_roi_1'],
                        data[data['result_roi_1'] == result][var1],
                        c=color, label=f'Result ROI 1: {result}')
        else:
            plt.scatter(data[data['result_roi_1'] == result]['time_roi_1'],
                        data[data['result_roi_1'] == result][var1],
                        c=color, label=f'Result ROI 1: {result}', marker='x')

    for result, color in zip([False, True, 'POT'], ['yellow', 'red', 'grey']):
        if result != 'POT':
            plt.scatter(data[data['result_roi_2'] == result]['time_roi_1'],
                        data[data['result_roi_2'] == result][var2],
                        c=color, label=f'Result ROI 2: {result}')
        else:
            plt.scatter(data[data['result_roi_2'] == result]['time_roi_1'],
                        data[data['result_roi_2'] == result][var2],
                        c=color, label=f'Result ROI 2: {result}', marker='x')

    plt.legend()
    plt.xlabel('Time (ms)')
    plt.ylabel('Sum Features')
    plt.title('Sum Features ROI 1 and ROI 2 over Time')

    plt.show()


# make_plot()


def make_histo():
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler

    var1 = 'psd_val_1'
    var2 = 'psd_val_2'

    data = pd.read_csv('./output/audit_data.csv')

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].hist(data[var1], bins=50, color='blue', alpha=0.7)
    axs[0].set_title(var1)
    axs[0].set_xlabel(var1)
    axs[0].set_ylabel('Frequency')
    axs[0].grid(True)

    axs[1].hist(data[var2], bins=50, color='red', alpha=0.7)
    axs[1].set_title(var2)
    axs[1].set_xlabel(var2)
    axs[1].set_ylabel('Frequency')
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


# make_histo()


def plot_kde(data, ax, title):
    kde = KernelDensity(bandwidth=0.5)
    kde.fit(data.values.reshape(-1, 1))
    x = np.linspace(data.min(), data.max(), 1000)
    log_dens = kde.score_samples(x.reshape(-1, 1))
    ax.plot(x, np.exp(log_dens), label=title)
    ax.set_title(f'Probability Distribution for {title}')
    ax.legend()
    return kde


def prob_dist():
    data = pd.read_csv('./output/audit_data2.csv')

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    mean1 = data['sum_features_roi_1'].mean()
    std1 = data['sum_features_roi_1'].std()
    axes[0].text(0.05, 0.95, f"Mean: {mean1:.2f}\nStd: {std1:.2f}", transform=axes[0].transAxes)

    mean2 = data['sum_features_roi_2'].mean()
    std2 = data['sum_features_roi_2'].std()
    axes[1].text(0.05, 0.95, f"Mean: {mean2:.2f}\nStd: {std2:.2f}", transform=axes[1].transAxes)

    plt.tight_layout()
    print('STD1:', std1, 'Mean1:', mean1)
    print('STD2:', std2, 'Mean2:', mean2)


def prob_dist2():
    data = pd.read_csv('./output/audit_data2.csv')
    scaler = MinMaxScaler()
    data[['sum_features_roi_1', 'sum_features_roi_2']] = scaler.fit_transform(
        data[['sum_features_roi_1', 'sum_features_roi_2']])

    # Plot the probability distribution
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data['sum_features_roi_1'], label='Sum Features ROI 1', color='blue', fill=True)
    sns.kdeplot(data['sum_features_roi_2'], label='Sum Features ROI 2', color='red', fill=True)

    plt.title('Probability Distribution of Sum Features')
    plt.xlabel('Normalized Sum Features')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True)
    plt.show()


def calc_mean_and_std2(data):
    stds = {}

    if data[0] == 'roi_1':
        stds['roi_1'] = np.std(data[1])
    elif data[0] == 'roi_2':
        stds['roi_2'] = np.std(data[1])

    return stds


def calc_mean_and_std(data):
    stds = []

    if data[0] == 'roi_1':
        stds.append(np.std(data[1], ddof=1))
    elif data[0] == 'roi_2':
        stds.append(np.std(data[1], ddof=1))
    return stds[-1]


def cal_bridge_time(roi_key, data):
    df = pd.DataFrame({'roi': [roi_key] * len(data), 'data': data})

    # Convert 'data' column into a DataFrame with two columns
    data_df = pd.DataFrame(df['data'].to_list(), columns=['value', 'result'])

    # Concatenate the new DataFrame with the original DataFrame
    result_df = pd.concat([df[['roi']], data_df], axis=1)

    # Check for consecutive occurrences of the value 1 and return 'Bridge'
    consecutive = 0
    for index, row in result_df.iterrows():
        if row['result'] == 1:
            consecutive += 1
            if consecutive == 3:
                return 'Bridge'
        elif row['result'] == -1:
            consecutive += 1
            if consecutive == 3:
                return 'Potential Bridge'

        else:
            consecutive = 0
