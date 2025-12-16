import requests

url = "http://localhost:8080/2015-03-31/functions/function/invocations"

customer = {
    "url":  "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"
    }

requests.post(url, json=customer, timeout=10)

result = requests.post(url, json=customer).json()

print(result)

