import pandas as pd
from flask import Flask, request, render_template
from flask_cors import CORS 
import pickle

app = Flask(__name__)
CORS(app)

with open('forecast_confirm_model.pckl', 'rb') as fin:
    con = pickle.load(fin)
    
with open('forecast_active_model.pckl', 'rb') as fin:
    act = pickle.load(fin)

with open('forecast_recovered_model.pckl', 'rb') as fin:
    rec = pickle.load(fin)

with open('forecast_death_model.pckl', 'rb') as fin:
    dead = pickle.load(fin)    

@app.route('/home')
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    val=[int(x) for x in request.form.values()]
    for v in val:
        val1=v
    
    #horizon = int(request.json['horizon'])
    future1 = con.make_future_dataframe(periods=val1)
    forecast1 = con.predict(future1)
    #data = pd.DataFrame.to_html(forecast2[['ds', 'yhat_upper']][-val1:])
    data1 = forecast1[['ds', 'yhat_upper']][-val1:]
    data1 = pd.DataFrame.rename(data1, columns={'ds':'Date','yhat_upper':'Predicted Total'})
    data1 = data1.set_index(['Date','Predicted Total'])
    
    #horizon = int(request.json['horizon'])
    future2 = act.make_future_dataframe(periods=val1)
    forecast2 = act.predict(future2)
    #data = pd.DataFrame.to_html(forecast2[['ds', 'yhat_upper']][-val1:])
    data2 = forecast2[['ds', 'yhat_upper']][-val1:]
    data2 = pd.DataFrame.rename(data2, columns={'ds':'Date','yhat_upper':'Predicted Active'})
    data2 = data2.set_index(['Date','Predicted Active'])    
     
    future3 = rec.make_future_dataframe(periods=val1)
    forecast3 = rec.predict(future3)
    #data = pd.DataFrame.to_html(forecast2[['ds', 'yhat_upper']][-val1:])
    data3 = forecast3[['ds', 'yhat_upper']][-val1:]
    data3 = pd.DataFrame.rename(data3, columns={'ds':'Date','yhat_upper':'Predicted Recovered'})
    data3 = data3.set_index(['Date','Predicted Recovered'])    
    
    future4 = dead.make_future_dataframe(periods=val1)
    forecast4 = dead.predict(future4)
    #data = pd.DataFrame.to_html(forecast2[['ds', 'yhat_upper']][-val1:])
    data4 = forecast4[['ds', 'yhat_upper']][-val1:]  
    #ret = data.to_json(orient='records', date_format='iso')
    data4 = pd.DataFrame.rename(data4, columns={'ds':'Date','yhat_upper':'Predicted Deaths'})
    data4 = data4.set_index(['Date','Predicted Deaths'])
    
    return render_template('predict.html', confirm=[data1.to_html(classes="Confirmed")], active=[data2.to_html(classes="Active")], deaths=[data4.to_html(classes="Deaths")], recover=[data3.to_html(classes="Recovered")], titles = ['na'])


if __name__ == "__main__":
    app.run(debug=True)
