import requests 

response = requests.get('http://rqk001.zapto.org:16707/hello/Orange')
print('response:', response)
print('code:', response.status_code)
print('json:', response.json())
