import datetime
import pickle
import time
import webbrowser
from flask import Flask, Response, json, request, app, jsonify, stream_with_context, url_for, render_template
from producer import Producer
from kafka import KafkaConsumer
import threading
from flask_sse import sse
from apscheduler.schedulers.background import BackgroundScheduler

import numpy as np
import pandas as pd

app=Flask(__name__)
# app.config["REDIS_URL"]="redis://localhost"
# app.register_blueprint(sse, url_prefix='/events')

#Load the model
regmodel=pickle.load(open('regmodel.pkl', 'rb'))
scalar=pickle.load(open('scaling.pkl', 'rb'))

request_topic = "prediction_topic"
result_topic = "prediction_result_topic"

producer = Producer(request_topic)

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict_api',methods=['POST'])
def predict_api():
    req_body=request.json
    data = []
    if req_body is not None:
        data=[float(req_body[k]) for k in req_body]

    producer.send(data)

    return jsonify({"status": "Data sent for processing"})

def request_consume():
    print("--->Consuming process started<----")
    kfkconsumer = KafkaConsumer(
        bootstrap_servers = ["localhost:9092"],
        group_id = "prediction_group",
        auto_offset_reset = "earliest",
        enable_auto_commit = False,
        consumer_timeout_ms = 1000,
        value_deserializer=lambda m: json.loads(m.decode('ascii'))
    )

    producer = Producer(result_topic)

    kfkconsumer.subscribe([request_topic])
    try:
        while True:
            message = kfkconsumer.poll(100)
            if not message:
                time.sleep(5)
                continue

            for topic_partition, messages in message.items():
                for msg in messages:
                    data = msg.value

                    final_input = scalar.transform(np.array(data).reshape(1,-1))
                    prediction = regmodel.predict(final_input)[0]

                    output = {"prediction": prediction}

                    producer.send(output)

                    print(f"Prediction: {prediction}")

            kfkconsumer.commit()
    except Exception as e:
        print(f"Error occured consuming: {str(e)}")
    finally:
        print("Closing consumer")
        kfkconsumer.close()

@app.route("/stream")
def send_response():
    def event_stream():
        consumer = KafkaConsumer(
            result_topic,
            bootstrap_servers=['localhost:9092'],
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            group_id='prediction_result_group',
            value_deserializer=lambda m: json.loads(m.decode('ascii'))
        )

        try:
            for message in consumer:
                data = message.value
                print(f"sending: {data}")
                yield f'data: {data}\n\n' 
        except Exception as e:
            print(f"Error: {e}")
        finally:
            consumer.close()

    resp = Response(event_stream(), content_type='text/event-stream')
    resp.headers["Cache-Control"] = "no-cache"
    resp.headers["Connection"] = "keep-alive"
    resp.headers["Access-Control-Allow-Origin"] = "*"

    return resp

# @app.route("/stream")
# def server_side_event():
#     consumer = KafkaConsumer(
#             result_topic,
#             bootstrap_servers=['localhost:9092'],
#             auto_offset_reset='earliest',
#             enable_auto_commit=True,
#             group_id='prediction_result_group',
#             value_deserializer=lambda m: json.loads(m.decode('ascii'))
#         )
    
#     for message in consumer:
#         data = message.value
#         print(f"sending: {data}")

#     sse.publish(data, type='prediction')
#     print("New Customer Time: ",datetime.datetime.now())
#     return "Message sent!"

if __name__=='__main__':
    # Automatically open the link in the default browser
    # webbrowser.open_new("http://127.0.0.1:5001")
    threading.Thread(target=request_consume, daemon=True).start()
    app.run(debug=True, use_reloader=False, port=5001)


producer.clean()