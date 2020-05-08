import pandas as pd

data = pd.read_csv('covid_19_data[2].csv')
india_data = data[data['Country/Region']=='India'].groupby('ObservationDate')['Confirmed','Recovered','Deaths'].sum()
df = india_data.groupby('ObservationDate').sum()['Confirmed'].reset_index()
df = df[['ObservationDate', 'Confirmed']].dropna()
df['ObservationDate'] = pd.to_datetime(df['ObservationDate'])
df = df.set_index('ObservationDate')
daily_df = df.resample('D').mean()
d_df = daily_df.reset_index().dropna()
d_df.columns = ['ds', 'y']
from fbprophet import Prophet
#Active cases
con = Prophet()
con.fit(d_df)
future = con.make_future_dataframe(periods=18)
forecast = con.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
#active pickled
import pickle
con.stan_backend.logger=None
with open('forecast_confirm_model.pckl','wb') as a:
 pickle.dump(con,a) #fout = a
 
