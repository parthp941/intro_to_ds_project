import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def allfeatureengineering(data):
    df = data.copy()

    # Convert Dates
    df['Date'] = pd.to_datetime(df['Date'])

    # Moving averages
    df['MovingAverage_50'] = df['Close'].rolling(50).mean() #average of 50 closing prices (rows) where 1 row = 1 day
    df['MovingAverage_200'] = df['Close'].rolling(200).mean() #average of 200 closing prices (rows) where 1 row = 1 day
    df['Daily_Return'] = df['Close'].pct_change() #finds daily returns (how much the price changed over one day. pct change: (new - old) / old)
    df['Volatility'] = df['Daily_Return'].rolling(window=20).std() #how much it changes for 20 days (low = good, high = bad)

    # find rsi
    change = df['Close'].diff() #finds the difference between today's close and yesterday's close price

    gain = change.clip(lower=0) # keeps only positive changes and turns negative ones into 0
    loss = -change.clip(upper=0) # keeps only negative changes and turns positives into 0 and makes losses positive

    average_gain = gain.rolling(14).mean() # finds the average gain from 14 days using moving average
    average_loss = loss.rolling(14).mean() # finds the average loss from 14 days using moving average

    rs = average_gain / average_loss # compares average gains to average losses

    df['RSI'] = 100 - (100 / (1 + rs)) # finds rsi


    # find macd
    exp_moving_average_12 = df['Close'].ewm(span=12).mean() #moving average for 12 days ewm is pandas method that weighs newer values more than older

    exp_moving_average_26 = df['Close'].ewm(span=26).mean() #moving average for 26 days


    df['MACD'] = exp_moving_average_12 - exp_moving_average_26

    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean() #average macd over past 9 days
    df['MACD_Historical'] = df['MACD'] - df['MACD_Signal'] #helps tell us the strength of what direction stock is moving in


    # BB (Bollinger Bands)
    bb_average = df['Close'].rolling(20).mean()
    bb_deviation = df['Close'].rolling(20).std()

    df['BB_Upper'] = bb_average + 2 * bb_deviation

    df['BB_Lower'] = bb_average - 2 * bb_deviation

    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']

    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / df['BB_Width']


    df['Target'] = df['Close'].shift(-1) #Target to predict tomorrow's closing price. sets tomorrows close price as todays target
    df['Target_Return'] = df['Close'].shift(-1) / df['Close'] - 1 #Target to predict tomorrow's percent change
    #Target to see if price goes up or down
    newcloseprice = df['Close'].shift(-1)
    df['Target_Direction'] = newcloseprice > df['Close'] #if newcloseprice is more than yesterday close price it goes up


    df = df.dropna()

    regime_features = df[["Volatility", "RSI", "BB_Width", "Daily_Return"]]
    scaled_regime_features = StandardScaler().fit_transform(regime_features)
    kmeans_model = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["Market_Regime"] = kmeans_model.fit_predict(scaled_regime_features)

    return df
