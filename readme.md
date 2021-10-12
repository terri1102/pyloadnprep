# pyloadnprep v.0.0.3

주가 데이터를 불러오고, 전처리하는 패키지입니다. 시계열 분석을 할 때 한 번에 데이터를 불러오고 전처리할 수 있게 구현하였습니다. 

## 공지
현재 Windows에서만 바로 사용이 가능하고, 다른 os의 경우 locale을 한국어로 설정해야 하는 문제가 있습니다.

## 설치
```bash
pip install pyloadnprep
```
## 기능확인
```python
print(help(pyloadnprep))
```
## 종목코드 확인
```python
from pyloadnprep.preprocessing import get_stockcode
```

## 데이터 불러오기

### 1. 거시 지표

### 2. 종목별 주가 데이터
```python
from pyloadnprep.preprocessing import make_dataset
make_dataset(stock_code, start_date='20000101', end_date=today)
```
OHLCV 데이터를 데이터 프레임으로 불러옵니다. 기본 설정은 2000년 1월 1일부터 오늘까지의 데이터를 불러옵니다.

### 3. 종목별 기본 지표
```python
from pyloadnprep.preprocessing import fundamental_analysis
fundamental_analysis(ticker, start_date='20000101', end_date=today)
```
종목별 BPS, PER, PBR, EPS, DIV,	DPS를 데이터 프레임으로 불러옵니다.

## 전처리

### 1. 종목별 기술 지표
```python
from pyloadnprep.preprocessing import technical_indices
technical_indices(df)
```
종목별 주가 데이터를 기술 지표들로 변환하여 데이터 프레임으로 반환합니다. 현재 지원하는 기술 지표들은 sma5, ema5, ema20, disparity, Slow_k, Slow_d, MACD, RSI, Diff입니다. 	

### 2. Fourier Transform


