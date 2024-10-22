from kafka import KafkaProducer
from kafka.errors import KafkaError
import json
from dotenv import load_dotenv
import os

load_dotenv()
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS")

class Producer:
    def __init__(self, topic):
        self.producer = KafkaProducer(
            bootstrap_servers = KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda m: json.dumps(m).encode('ascii')
        )
        self.topic = topic
    
    def _on_success(self, metadata):
        print(f"Message produced to topic '{metadata.topic}' at offset {metadata.offset}")
    
    def _on_error(self, e):
        print(f"Error sending message: {e}")

    ''' make sure to call method clean after sending'''
    def send(self, json_data):
        future = self.producer.send(self.topic, json_data)
        future.add_callback(self._on_success)
        future.add_callback(self._on_error)
    
    def clean(self):
        print("Closing producer")
        self.producer.flush()
        self.producer.close()