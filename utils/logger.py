
import requests

def send_log(filename, msg):

    requests.post("https://cv.doky.space", json={
        "log": msg,
        "name": filename,
    })