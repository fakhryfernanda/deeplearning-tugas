import time
import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
from alpha_vantage.techindicators import TechIndicators
import matplotlib.pyplot as plt
from PIL import Image
import os


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

        # time.sleep(12)

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

        #time.sleep(12)

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

        #time.sleep(12)

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

        #time.sleep(12)

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

        #time.sleep(12)

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

        #time.sleep(12)

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

        #time.sleep(12)

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

        #time.sleep(12)

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

        #time.sleep(12)

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

        #time.sleep(12)

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

        #time.sleep(12)

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

        #time.sleep(12)

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

        #time.sleep(12)

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

        #time.sleep(12)

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

        #time.sleep(12)

    return df


#### CREATE LABELS FROM CLOSE PRICE ####


def create_labels(df, window_size=11):
    '''
    Data is labeled as per algorithm in reference paper.
    BUY = 1, SELL = 0, HOLD = 2

    input: df - data
    output: labels as np array of integers, size=len(df)-window_size+1
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


#### RESHAPE DATA AS IMAGE ####
def reshape_data(x, width, height):
    x_temp = np.zeros((len(x), height, width))

    for i in range(x.shape[0]):
        x_temp[i] = np.reshape(x[i], (height, width))

    return x_temp


def show_images(rows, columns, path):
    w = 10
    h = 10
    fig = plt.figure(figsize=(10, 10))
    files = os.listdir(path)
    for i in range(1, columns * rows + 1):
        index = np.random.randint(len(files))
        img = np.asarray(Image.open(os.path.join(path, files[index])))
        fig.add_subplot(rows, columns, i)
        plt.title(files[i], fontsize=10)
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        plt.imshow(img)
    plt.show()


# def save_array_as_images(x, img_width, img_height, path, file_names):
#     if os.path.exists(path):
#         shutil.rmtree(path)
#         print("deleted old files")

#     os.makedirs(path)
#     print("Image Directory created", path)
#     x_temp = np.zeros((len(x), img_height, img_width))
#     print("saving images...")
#     stime = time.time()
#     for i in tqdm(range(x.shape[0])):
#         x_temp[i] = np.reshape(x[i], (img_height, img_width))
#         img = Image.fromarray(x_temp[i], 'RGB')
#         img.save(os.path.join(path, str(file_names[i]) + '.png'))

#     print_time("Images saved at " + path, stime)
#     return x_temp
