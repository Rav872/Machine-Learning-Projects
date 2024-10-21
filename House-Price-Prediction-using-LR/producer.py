from kafka import KafkaProducer
from kafka.errors import KafkaError
import json

class Producer:
    def __init__(self, topic):
        self.producer = KafkaProducer(
            bootstrap_servers = "localhost:9092",
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