import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from alpha_vantage.techindicators import TechIndicators
from sklearn.utils import compute_class_weight
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt


#### 15 TECHNICAL INDICATORS ####
def get_ema(ticker, intervals, start, end):
    print("--- Obtaining EMA ---")
    ti = TechIndicators(key="GRBM46BDRML13M92", output_format="pandas")

    ema, _ = ti.get_ema(symbol=ticker, time_period=intervals[0])
    df = ema[start:end]
    df = df.rename(columns={"EMA": f"ema_{intervals[0]}"})

    for i, period in enumerate(intervals[1:]):
        ema, _ = ti.get_ema(symbol=ticker, time_period=period)
        df = pd.concat([df, ema[start:end]], axis=1)
        df = df.rename(columns={"EMA": f"ema_{period}"})

        time.sleep(12)

    return df


def get_sma(ticker, intervals, start, end):
    print("--- Obtaining SMA ---")
    ti = TechIndicators(key="GRBM46BDRML13M92", output_format="pandas")

    sma, _ = ti.get_sma(symbol=ticker, time_period=intervals[0])
    df = sma[start:end]
    df = df.rename(columns={"SMA": f"sma_{intervals[0]}"})

    for i, period in enumerate(intervals[1:]):
        sma, _ = ti.get_sma(symbol=ticker, time_period=period)
        df = pd.concat([df, sma[start:end]], axis=1)
        df = df.rename(columns={"SMA": f"sma_{period}"})

        time.sleep(12)

    return df


def get_wma(ticker, intervals, start, end):
    print("--- Obtaining WMA ---")
    ti = TechIndicators(key="GRBM46BDRML13M92", output_format="pandas")

    wma, _ = ti.get_wma(symbol=ticker, time_period=intervals[0])
    df = wma[start:end]
    df = df.rename(columns={"WMA": f"wma_{intervals[0]}"})

    for i, period in enumerate(intervals[1:]):
        wma, _ = ti.get_wma(symbol=ticker, time_period=period)
        df = pd.concat([df, wma[start:end]], axis=1)
        df = df.rename(columns={"WMA": f"wma_{period}"})

        time.sleep(12)

    return df


def get_trima(ticker, intervals, start, end):
    print("--- Obtaining TRIMA ---")
    ti = TechIndicators(key="GRBM46BDRML13M92", output_format="pandas")

    trima, _ = ti.get_trima(symbol=ticker, time_period=intervals[0])
    df = trima[start:end]
    df = df.rename(columns={"TRIMA": f"trima_{intervals[0]}"})

    for i, period in enumerate(intervals[1:]):
        trima, _ = ti.get_trima(symbol=ticker, time_period=period)
        df = pd.concat([df, trima[start:end]], axis=1)
        df = df.rename(columns={"TRIMA": f"trima_{period}"})

        time.sleep(12)

    return df


def get_t3(ticker, intervals, start, end):
    print("--- Obtaining T3 ---")
    ti = TechIndicators(key="GRBM46BDRML13M92", output_format="pandas")

    t3, _ = ti.get_t3(symbol=ticker, time_period=intervals[0])
    df = t3[start:end]
    df = df.rename(columns={"T3": f"t3_{intervals[0]}"})

    for i, period in enumerate(intervals[1:]):
        t3, _ = ti.get_t3(symbol=ticker, time_period=period)
        df = pd.concat([df, t3[start:end]], axis=1)
        df = df.rename(columns={"T3": f"t3_{period}"})

        time.sleep(12)

    return df


def get_rsi(ticker, intervals, start, end):
    print("--- Obtaining RSI ---")
    ti = TechIndicators(key="GRBM46BDRML13M92", output_format="pandas")

    rsi, _ = ti.get_rsi(symbol=ticker, time_period=intervals[0])
    df = rsi[start:end]
    df = df.rename(columns={"RSI": f"rsi_{intervals[0]}"})

    for i, period in enumerate(intervals[1:]):
        rsi, _ = ti.get_rsi(symbol=ticker, time_period=period)
        df = pd.concat([df, rsi[start:end]], axis=1)
        df = df.rename(columns={"RSI": f"rsi_{period}"})

        time.sleep(12)

    return df


def get_willr(ticker, intervals, start, end):
    print("--- Obtaining WILLR ---")
    ti = TechIndicators(key="GRBM46BDRML13M92", output_format="pandas")

    willr, _ = ti.get_willr(symbol=ticker, time_period=intervals[0])
    df = willr[start:end]
    df = df.rename(columns={"WILLR": f"willr_{intervals[0]}"})

    for i, period in enumerate(intervals[1:]):
        willr, _ = ti.get_willr(symbol=ticker, time_period=period)
        df = pd.concat([df, willr[start:end]], axis=1)
        df = df.rename(columns={"WILLR": f"willr_{period}"})

        time.sleep(12)

    return df


def get_adx(ticker, intervals, start, end):
    print("--- Obtaining ADX ---")
    ti = TechIndicators(key="GRBM46BDRML13M92", output_format="pandas")

    adx, _ = ti.get_adx(symbol=ticker, time_period=intervals[0])
    df = adx[start:end]
    df = df.rename(columns={"ADX": f"adx_{intervals[0]}"})

    for i, period in enumerate(intervals[1:]):
        adx, _ = ti.get_adx(symbol=ticker, time_period=period)
        df = pd.concat([df, adx[start:end]], axis=1)
        df = df.rename(columns={"ADX": f"adx_{period}"})

        time.sleep(12)

    return df


def get_mom(ticker, intervals, start, end):
    print("--- Obtaining MOM ---")
    ti = TechIndicators(key="GRBM46BDRML13M92", output_format="pandas")

    mom, _ = ti.get_mom(symbol=ticker, time_period=intervals[0])
    df = mom[start:end]
    df = df.rename(columns={"MOM": f"mom_{intervals[0]}"})

    for i, period in enumerate(intervals[1:]):
        mom, _ = ti.get_mom(symbol=ticker, time_period=period)
        df = pd.concat([df, mom[start:end]], axis=1)
        df = df.rename(columns={"MOM": f"mom_{period}"})

        time.sleep(12)

    return df


def get_cci(ticker, intervals, start, end):
    print("--- Obtaining CCI ---")
    ti = TechIndicators(key="GRBM46BDRML13M92", output_format="pandas")

    cci, _ = ti.get_cci(symbol=ticker, time_period=intervals[0])
    df = cci[start:end]
    df = df.rename(columns={"CCI": f"cci_{intervals[0]}"})

    for i, period in enumerate(intervals[1:]):
        cci, _ = ti.get_cci(symbol=ticker, time_period=period)
        df = pd.concat([df, cci[start:end]], axis=1)
        df = df.rename(columns={"CCI": f"cci_{period}"})

        time.sleep(12)

    return df


def get_cmo(ticker, intervals, start, end):
    print("--- Obtaining CMO ---")
    ti = TechIndicators(key="GRBM46BDRML13M92", output_format="pandas")

    cmo, _ = ti.get_cmo(symbol=ticker, time_period=intervals[0])
    df = cmo[start:end]
    df = df.rename(columns={"CMO": f"cmo_{intervals[0]}"})

    for i, period in enumerate(intervals[1:]):
        cmo, _ = ti.get_cmo(symbol=ticker, time_period=period)
        df = pd.concat([df, cmo[start:end]], axis=1)
        df = df.rename(columns={"CMO": f"cmo_{period}"})

        time.sleep(12)

    return df


def get_roc(ticker, intervals, start, end):
    print("--- Obtaining ROC ---")
    ti = TechIndicators(key="GRBM46BDRML13M92", output_format="pandas")

    roc, _ = ti.get_roc(symbol=ticker, time_period=intervals[0])
    df = roc[start:end]
    df = df.rename(columns={"ROC": f"roc_{intervals[0]}"})

    for i, period in enumerate(intervals[1:]):
        roc, _ = ti.get_roc(symbol=ticker, time_period=period)
        df = pd.concat([df, roc[start:end]], axis=1)
        df = df.rename(columns={"ROC": f"roc_{period}"})

        time.sleep(12)

    return df


def get_mfi(ticker, intervals, start, end):
    print("--- Obtaining MFI ---")
    ti = TechIndicators(key="GRBM46BDRML13M92", output_format="pandas")

    mfi, _ = ti.get_mfi(symbol=ticker, time_period=intervals[0])
    df = mfi[start:end]
    df = df.rename(columns={"MFI": f"mfi_{intervals[0]}"})

    for i, period in enumerate(intervals[1:]):
        mfi, _ = ti.get_mfi(symbol=ticker, time_period=period)
        df = pd.concat([df, mfi[start:end]], axis=1)
        df = df.rename(columns={"MFI": f"mfi_{period}"})

        time.sleep(12)

    return df


def get_trix(ticker, intervals, start, end):
    print("--- Obtaining TRIX ---")
    ti = TechIndicators(key="GRBM46BDRML13M92", output_format="pandas")

    trix, _ = ti.get_trix(symbol=ticker, time_period=intervals[0])
    df = trix[start:end]
    df = df.rename(columns={"TRIX": f"trix_{intervals[0]}"})

    for i, period in enumerate(intervals[1:]):
        trix, _ = ti.get_trix(symbol=ticker, time_period=period)
        df = pd.concat([df, trix[start:end]], axis=1)
        df = df.rename(columns={"TRIX": f"trix_{period}"})

        time.sleep(12)

    return df


def get_dx(ticker, intervals, start, end):
    print("--- Obtaining DX ---")
    ti = TechIndicators(key="GRBM46BDRML13M92", output_format="pandas")

    dx, _ = ti.get_dx(symbol=ticker, time_period=intervals[0])
    df = dx[start:end]
    df = df.rename(columns={"DX": f"dx_{intervals[0]}"})

    for i, period in enumerate(intervals[1:]):
        dx, _ = ti.get_dx(symbol=ticker, time_period=period)
        df = pd.concat([df, dx[start:end]], axis=1)
        df = df.rename(columns={"DX": f"dx_{period}"})

        time.sleep(12)

    return df


# Create labels from close price
def create_labels(df, window_size=11):
    '''
    Data is labeled as per algorithm in reference paper.
    0 - SELL
    1 - BUY
    2 - HOLD

    Input: df - data
    Output: labels - np array of integers, size=len(df)-window_size+1
    '''

    row_counter = 0
    total_rows = len(df)
    labels = np.zeros(total_rows)

    print("--- Creating labels ---")
    pbar = tqdm(total_rows)

    while row_counter < total_rows:
        if row_counter >= window_size - 1:
            window_begin = row_counter - (window_size - 1)
            window_end = row_counter
            window_middle = int((window_begin + window_end) / 2)

            min_ = np.inf
            min_index = -1
            max_ = -np.inf
            max_index = -1

            for i in range(window_begin, window_end + 1):
                price = df.iloc[i]["close"]
                if price < min_:
                    min_ = price
                    min_index = i
                if price > max_:
                    max_ = price
                    max_index = i

            if max_index == window_middle:
                labels[window_middle] = 0
            elif min_index == window_middle:
                labels[window_middle] = 1
            else:
                labels[window_middle] = 2

        row_counter = row_counter + 1
        pbar.update(1)

    pbar.close()
    return labels


# Reshape data into 2D each
def reshape_data(x, width, height):
    x_temp = np.zeros((len(x), height, width))

    for i in range(x.shape[0]):
        x_temp[i] = np.reshape(x[i], (height, width))

    return x_temp


# Calculate sample weights
def get_sample_weights(y):
    """
    Calculate the sample weights based on class weights. Used for models with
    imbalanced data and one hot encoding prediction.

    Input: y - class labels as integers
    """

    y = y.astype(int)  # compute_class_weight needs int labels
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)

    print(f"Class weights {class_weights}")
    print(f"Value counts: {np.unique(y, return_counts=True)}")

    sample_weights = y.copy().astype(float)

    for i in np.unique(y):
        sample_weights[sample_weights == i] = class_weights[i]

    return sample_weights


# Calculate F1-Score of model
def f1_score(y_true, y_pred):

    def recall(y_true, y_pred):

        # mistake: y_pred of 0.3 is also considered 1
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())

        return recall

    def precision(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())

        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)

    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


# Plot metrics of neural network model
def plot_history(history):
    # Plot loss
    plt.plot([i + 1 for i in range(len(history["loss"]))], history['loss'])
    plt.plot([i + 1 for i in range(len(history["loss"]))], history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    # Plot accuracy
    plt.plot([i + 1 for i in range(len(history["loss"]))], history['accuracy'])
    plt.plot([i + 1 for i in range(len(history["loss"]))],
             history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    # Plot f1-score
    plt.plot([i + 1 for i in range(len(history["loss"]))], history['f1_score'])
    plt.plot([i + 1 for i in range(len(history["loss"]))],
             history['val_f1_score'])
    plt.title('Model F1-Score')
    plt.ylabel('F1-Score')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
