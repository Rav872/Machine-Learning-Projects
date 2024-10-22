import datetime
import pickle
import time
import webbrowser
from flask import Flask, Response, json, request, jsonify, render_template
from producer import Producer
from kafka import KafkaConsumer
import threading
from flask_sse import sse
from apscheduler.schedulers.background import BackgroundScheduler

import numpy as np
import pandas as pd

app = Flask(__name__)
app.config["REDIS_URL"] = "redis://localhost"
app.register_blueprint(sse, url_prefix='/stream')

# Load the model
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))

request_topic = "prediction_topic"
result_topic = "prediction_result_topic"

producer = Producer(request_topic)
prediction_consumer = None

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    req_body = request.json
    data = []
    if req_body is not None:
        data = [float(req_body[k]) for k in req_body]

    producer.send(data)
    return jsonify({"status": "Data sent for processing"})

def request_consume():
    print("--->Consuming process started<----")
    kfkconsumer = KafkaConsumer(
        bootstrap_servers=["localhost:9092"],
        group_id="prediction_group",
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        consumer_timeout_ms=1000,
        value_deserializer=lambda m: json.loads(m.decode('ascii'))
    )

    result_producer = Producer(result_topic)

    kfkconsumer.subscribe([request_topic])
    try:
        while True:
            message = kfkconsumer.poll(100)
            if not message:
                time.sleep(1)
                continue

            for topic_partition, messages in message.items():
                for msg in messages:
                    data = msg.value
                    final_input = scalar.transform(np.array(data).reshape(1,-1))
                    prediction = regmodel.predict(final_input)[0]
                    
                    output = {"prediction": prediction}
                    result_producer.send(output)
                    print(f"Prediction made: {prediction}")

            kfkconsumer.commit()
    except Exception as e:
        print(f"Error occurred consuming: {str(e)}")
    finally:
        print("Closing consumer")
        kfkconsumer.close()

def initialize_prediction_consumer():
    global prediction_consumer
    if prediction_consumer is None:
        prediction_consumer = KafkaConsumer(
            result_topic,
            bootstrap_servers=['localhost:9092'],
            auto_offset_reset='latest',
            enable_auto_commit=True,
            group_id='prediction_result_group',
            consumer_timeout_ms=500,
            value_deserializer=lambda m: json.loads(m.decode('ascii'))
        )

def prediction_response():
    global prediction_consumer
    
    try:
        if prediction_consumer is None:
            initialize_prediction_consumer()
            
        messages = prediction_consumer.poll(timeout_ms=100)
        
        if messages:
            with app.app_context():
                for topic_partition, msgs in messages.items():
                    for msg in msgs:
                        data = msg.value
                        print(f"Publishing prediction: {data}")
                        sse.publish(data, type='prediction')
                        print("New Prediction Time: ", datetime.datetime.now())
        
    except Exception as e:
        print(f"Error in prediction_response: {str(e)}")
        prediction_consumer = None

def scheduler_thread():
    scheduler = BackgroundScheduler(daemon=True)
    scheduler.add_job(
        prediction_response,
        'interval',
        seconds=2,  
        misfire_grace_time=1, 
        coalesce=True,
        max_instances=1,
        id='prediction_job'
    )
    scheduler.start()

if __name__ == '__main__':
    threading.Thread(target=request_consume, daemon=True).start()
    threading.Thread(target=scheduler_thread, daemon=True).start()
    app.run(debug=False, use_reloader=False, port=5001)
    
    if prediction_consumer:
        prediction_consumer.close()
    producer.clean()