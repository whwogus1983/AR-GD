from flask import Flask, request
from prometheus_client import Counter, Histogram, generate_latest, REGISTRY
import random
import time

app = Flask(__name__)

REQUEST_COUNT = Counter('request_count', 'Total request count')
REQUEST_LATENCY = Histogram('request_latency_seconds', 'Request latency in seconds')

@app.route('/')
@REQUEST_LATENCY.time()
@REQUEST_COUNT.count_exceptions()
def hello():
    time.sleep(random.uniform(0.1, 0.5))
    return 'Hello, World!'

@app.route('/metrics')
def metrics():
    return generate_latest(REGISTRY)

if __name__ == '__main__':
    app.run(debug=True)
