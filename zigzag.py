import numpy as np
import plotly.graph_objects as go
import pandas as pd
import os

PEAK, VALLEY = 1, -1

def _identify_initial_pivot(X, up_thresh, down_thresh):
    """Quickly identify the X[0] as a peak or valley."""
    x_0 = X[0]
    max_x = x_0
    max_t = 0
    min_x = x_0
    min_t = 0
    up_thresh += 1
    down_thresh += 1

    for t in range(1, len(X)):
        x_t = X[t]

        if x_t / min_x >= up_thresh:
            return VALLEY if min_t == 0 else PEAK

        if x_t / max_x <= down_thresh:
            return PEAK if max_t == 0 else VALLEY

        if x_t > max_x:
            max_x = x_t
            max_t = t

        if x_t < min_x:
            min_x = x_t
            min_t = t

    t_n = len(X)-1
    return VALLEY if x_0 < X[t_n] else PEAK

def peak_valley_pivots_candlestick(close, high, low, up_thresh, down_thresh):
    """
    Finds the peaks and valleys of a series of HLC (open is not necessary).
    TR: This is modified peak_valley_pivots function in order to find peaks and valleys for OHLC.
    Parameters
    ----------
    close : This is series with closes prices.
    high : This is series with highs  prices.
    low : This is series with lows prices.
    up_thresh : The minimum relative change necessary to define a peak.
    down_thesh : The minimum relative change necessary to define a valley.
    Returns
    -------
    an array with 0 indicating no pivot and -1 and 1 indicating valley and peak
    respectively
    Using Pandas
    ------------
    For the most part, close, high and low may be a pandas series. However, the index must
    either be [0,n) or a DateTimeIndex. Why? This function does X[t] to access
    each element where t is in [0,n).
    The First and Last Elements
    ---------------------------
    The first and last elements are guaranteed to be annotated as peak or
    valley even if the segments formed do not have the necessary relative
    changes. This is a tradeoff between technical correctness and the
    propensity to make mistakes in data analysis. The possible mistake is
    ignoring data outside the fully realized segments, which may bias analysis.
    """
    if down_thresh > 0:
        raise ValueError('The down_thresh must be negative.')

    initial_pivot = _identify_initial_pivot(close, up_thresh, down_thresh)

    t_n = len(close)
    pivots = np.zeros(t_n, dtype='i1')
    pivots[0] = initial_pivot

    # Adding one to the relative change thresholds saves operations. Instead
    # of computing relative change at each point as x_j / x_i - 1, it is
    # computed as x_j / x_1. Then, this value is compared to the threshold + 1.
    # This saves (t_n - 1) subtractions.
    # up_thresh += 1
    # down_thresh += 1

    trend = -initial_pivot
    last_pivot_t = 0
    if(trend == -1):
        last_pivot_x = high[0]
    else:
        last_pivot_x = low[0]
    for t in range(1, len(close)):
        xl = low[t]
        xh = high[t]
        rl = 1- last_pivot_x / xl
        rh = xh / last_pivot_x -1
        if trend == -1:
            # x = low[t]
            # r = x / last_pivot_x
            if rh >= up_thresh:
                pivots[last_pivot_t] = trend#
                trend = 1
                #last_pivot_x = x
                last_pivot_x = high[t]
                last_pivot_t = t
            elif xl < last_pivot_x:
                last_pivot_x = xl
                last_pivot_t = t
        else:
            # x = high[t]
            # r = x / last_pivot_x
            if rl <= down_thresh:
                pivots[last_pivot_t] = trend
                trend = -1
                #last_pivot_x = x
                last_pivot_x = low[t]
                last_pivot_t = t
            elif xh > last_pivot_x:
                last_pivot_x = xh
                last_pivot_t = t



    if last_pivot_t == t_n-1:
        pivots[last_pivot_t] = trend
    elif pivots[t_n-1] == 0:
        pivots[t_n-1] = trend

    return pivots

csv_filename = []
for root, dirs, files in os.walk('/Users/tutu/coding/bt_stock/testdata/day/'):
    csv_filename = files
print(csv_filename)
# assert 0
for csvname in csv_filename:
    df = pd.read_csv('/Users/tutu/coding/bt_stock/testdata/day/' + csvname)



    pivots = peak_valley_pivots_candlestick(df.close, df.high, df.low ,.08,-.08)
    df['Pivots'] = pivots
    df['P_Price'] = np.nan  # This line clears old pivot prices
    df.loc[df['Pivots'] == 1, 'P_Price'] = df.high
    df.loc[df['Pivots'] == -1, 'P_Price'] = df.low

    # zg_df = df.loc[df['P_Price'].notnull()]
    # zg_df.to_csv('./slice/000001/zg.csv')
    # print(zg_df[['Pivots','P_Price']].iloc[1:])
    for i, p in enumerate(df['P_Price']):
        # print(p)
        if not np.isnan(p) and df['Pivots'].iloc[i] == -1:
            for y in (df[['Pivots','P_Price']].iloc[i+1:].itertuples()):
                if(getattr(y,'Pivots') == -1):
                    if(getattr(y,'P_Price') < p):
                        break
                    if (abs(getattr(y,'P_Price') - p) <= 0.05 or 0 < (p - getattr(y,'P_Price'))/p <= 0.03) and getattr(y,'Pivots') == df['Pivots'].iloc[i]:
                        path = './testdata/' + csvname[0:6] + '/'
                        if not os.path.exists(path):
                            os.system(r"mkdir {}".format(path))
                        df.iloc[i:getattr(y,'Index')+1].to_csv(path + str(i) + str(getattr(y,'Index')) + '.csv')
                        print(p,getattr(y,'P_Price'))
    # assert 0
    fig = go.Figure(data=[go.Candlestick(x=df['datetime'],

                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'])])


    df_diff = df['P_Price'].dropna().diff().copy()

assert 0
fig.add_trace(
    go.Scatter(mode = "lines+markers",
        x=df['datetime'],
        y=df["P_Price"]
    ))

fig.update_layout(
    autosize=False,
    width=1000,
    height=800,)

fig.add_trace(go.Scatter(x=df['datetime'],
                         y=df['P_Price'].interpolate(),
                         mode = 'lines',
                         line = dict(color='black')))


def annot(value):
    if np.isnan(value):
        return ''
    else:
        return value
    

j = 0
for i, p in enumerate(df['P_Price']):
    if not np.isnan(p):

        
        fig.add_annotation(dict(font=dict(color='rgba(0,0,200,0.8)',size=12),
                                        x=df['datetime'].iloc[i],
                                        y=p,
                                        showarrow=False,
                                        text=annot(round(abs(df_diff.iloc[j]),3)),
                                        textangle=0,
                                        xanchor='right',
                                        xref="x",
                                        yref="y"))
        j = j + 1

        
fig.update_xaxes(type='category')
fig.show()