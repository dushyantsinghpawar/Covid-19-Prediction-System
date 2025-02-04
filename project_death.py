import pandas as pd

data = pd.read_csv('covid_19_data.csv')
india_data = data[data['Country/Region']=='India'].groupby('ObservationDate')['Confirmed','Recovered','Deaths'].sum()
india_data['Active']=india_data['Confirmed']-india_data['Deaths']-india_data['Recovered']
df = india_data.groupby('ObservationDate').sum()['Deaths'].reset_index()
df = df[['ObservationDate', 'Deaths']].dropna()
df['ObservationDate'] = pd.to_datetime(df['ObservationDate'])
df = df.set_index('ObservationDate')
daily_df = df.resample('D').mean()
d_df = daily_df.reset_index().dropna()
d_df.columns = ['ds', 'y']
from fbprophet import Prophet
#Active cases
dead = Prophet()
dead.fit(d_df)
future = dead.make_future_dataframe(periods=18)
forecast = dead.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
#active pickled
import pickle
dead.stan_backend.logger=None
with open('forecast_death_model.pckl','wb') as a:
 pickle.dump(dead,a) #fout = a
 

