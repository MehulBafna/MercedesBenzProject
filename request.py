import requests
url = 'http://127.0.0.1:5000/predict_api'
r = requests.post(url,json={'Model Year':2019,'Make':1,'Model':2,'Electric Vehicle Type':1,'Clean Alternative Fuel Vehicle (CAFV) Eligibility':2,'Electric Range':125,'Base MSRP':30000,'Legislative District':12.0})
print(r.json())