#전처리 스크립트
#api에 입력할 값: 회사명 or ticker , 시작일, 종료일
import locale
locale.setlocale(locale.LC_TIME,'ko_KR.UTF-8')

import datetime
now = datetime.datetime.now()
today = now.strftime("%Y%m%d")
start_date = '20000101'
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
#from tensorflow.keras.models import Model
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


from pykrx import stock
   #종목코드 구하기
def get_stockcode(stockname): #stockname:string

#stock_code = stock.get_market_ticker_list(date=today, market="ALL")

    ticker_name = {}
    for ticker in stock.get_market_ticker_list():
            종목 = stock.get_market_ticker_name(ticker)
            ticker_name[종목] = ticker
    stock_code = ticker_name[stockname]
    return stock_code

#데이터프레임 불러오기
def make_dataset(stock_code, start_date='20000101', end_date=today):
    single_dataframe = stock.get_market_ohlcv_by_date(start_date, end_date, stock_code)
    return single_dataframe


# 전처리
# 1) technical_indices: df1
def technical_indices(df):
    df = df.astype('float')

    #5일 단순이동평균(sma)
    df['sma5'] = df['종가'].rolling(window=5, min_periods=1).mean()

    #5일 지수이동평균(ema)
    df['ema5'] = df['종가'].ewm(5).mean()

    #20일 지수이동평균
    df['ema20'] = df['종가'].ewm(20).mean()

    #이격도
    df['disparity'] = df['종가'] - df['sma5']
    
    #Stochastic K, D : Fast %K의 m기간 이동평균(SMA)
    df['fast_k'] = ((df['종가'] - df['저가'].rolling(5).min()) / (df['고가'].rolling(5).max() - df['저가'].rolling(5).min())) * 100
    df['Slow_k'] = df['fast_k'].rolling(3).mean()
    df['Slow_d'] = df['Slow_k'].rolling(3).mean()

    #MACD
    df['EMAFast'] = df['종가'].ewm( span = 5, min_periods = 4).mean()
    df['EMASlow'] = df['종가'].ewm( span = 20, min_periods = 19).mean()
    df['MACD'] = df['EMAFast'] - df['EMASlow']
    #df['MACDSignal'] = df['MACD'].ewm( span = 9, min_periods = 8).mean()
    #df['MACDDiff'] = df['MACD'] - df['MACDSignal']

    #RSI
    delta = df['종가'].diff(5)
    delta = delta.dropna()

    up = delta.copy()
    down = delta.copy()

    up[up<0] = 0
    down[down>0] = 0

    df['up'] = up
    df['down'] = down

    AVG_Gain = df['up'].rolling(window = 5).mean()
    AVG_Loss = abs(df['down'].rolling(window = 5).mean())
    RS = AVG_Gain/AVG_Loss

    RSI = 100.0 - (100.0/(1+RS))
    df['RSI'] = RSI 

    df['Diff'] =  np.r_[0, np.diff(df['종가'])]

    df = df.drop(columns=['fast_k', 'EMAFast', 'EMASlow', 'up', 'down'])
    df = df.dropna()

    df = df.reset_index() #time_series_split할 때 필요


    return df #주의할 점: 이동평균 구하는 것 때문에 시작 일이 바뀜!


#2) fundamental analysis : df2
#위의 technical_indices로 나온 df와 별개로 만들기
def fundamental_analysis(ticker, start_date='20000101', end_date=today):
    df = stock.get_market_fundamental_by_date(fromdate=start_date, todate=end_date, ticker=ticker)
    df = df.reset_index()
    return df

#3)fourier transform : 장,단기 추세
def fourier_transform(df):
    df = df.reset_index() #make_dataset에서 하면 technical~에서 오류
    data_ft = df[['날짜','종가']]
    close_fft = np.fft.fft(np.asarray(data_ft['종가'].tolist()))
    fft_df = pd.DataFrame({'fft':close_fft})
    fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
    fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))
    fft_list = np.asarray(fft_df['fft'].tolist())
    ft_df = []
    for num_ in [3, 6, 9]:
        fft_list_m10= np.copy(fft_list); fft_list_m10[num_:-num_]=0
        ft_df.append(np.real(fft_list_m10)) #그래프로 그려보니 실수부분만 써도 비슷한 추세 보임
    ft_df = pd.DataFrame(ft_df, index=['3','6','9']).T
    ft_df = pd.concat([df[['날짜']],ft_df], axis=1)
    return ft_df

#market index
def get_market_index(start_date,end_date):
    market1 = stock.get_market_ohlcv_by_date(start_date, today, "152380").reset_index()[['날짜','종가']].rename(columns={'종가':'KODEX국채선물10년'}) #KODEX 국채선물10년
    market2 = stock.get_market_ohlcv_by_date(start_date, today, "114820").reset_index()[['날짜','종가']].rename(columns={'종가':'TIGER 국채3년'})   #TIGER 국채3년
    market4 = stock.get_market_ohlcv_by_date(start_date, today, "138230").reset_index()[['날짜','종가']].rename(columns={'종가':'KOSEF 미국달러선물'})  #KOSEF 미국달러선물
    market_index = pd.merge(market1, market2, how='inner',on=['날짜'])
    market_index = pd.merge(market_index, market4, how='inner',on=['날짜'])
    return market_index

#autoencoder
def stacked_auto_encoder(df):
    data = df[['종가']]


def preprocess_all_options(companyname):
    stock_code = get_stockcode(companyname)
    data1 = make_dataset(stock_code, start_date=start_date, end_date=today)
    df1 = technical_indices(data1)
    df2 = fundamental_analysis(stock_code, start_date=start_date, end_date=today)
    df3 = fourier_transform(data1)
    df4 = get_market_index(start_date=start_date, end_date=today)
    final_df1 = pd.merge(df1,df2, how='inner',on=['날짜'])
    final_df2 = pd.merge(final_df1,df3, how='inner',on=['날짜'])
    final_df = pd.merge(final_df2, df4, how='inner',on=['날짜'])
    final_df.to_csv('{}.csv'.format(stock_code),index=False)
    return final_df


if __name__ == '__main__':
   #시가 총액 6위: 삼성전자, SK하이닉스, NAVER, (삼성전자우) 카카오, LG화학
    #market_capital6 = ['삼성전자', 'SK하이닉스', 'NAVER', '카카오', 'LG화학']
    market_capital6 = ['삼성전자','SK하이닉스', 'NAVER', '카카오', 'LG화학']
    for x in market_capital6:
        stock_code1 = get_stockcode(x)
        data1 = make_dataset(stock_code1, start_date=start_date, end_date=today)
        df1 = technical_indices(data1)
        df2 = fundamental_analysis(stock_code1, start_date=start_date, end_date=today)
        df3 = fourier_transform(data1)
        df4 = get_market_index(start_date=start_date, end_date=today)
        final_df1 = pd.merge(df1,df2, how='inner',on=['날짜'])
        final_df2 = pd.merge(final_df1,df3, how='inner',on=['날짜'])
        final_df = pd.merge(final_df2, df4, how='inner',on=['날짜'])
        final_df = final_df.rename(columns={"날짜":'date','종가': 'price','거래량':'volume'})
        final_df.to_csv('{}.csv'.format(stock_code1),index=False)

    


