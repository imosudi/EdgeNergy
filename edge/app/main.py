import threading
import json
import time
from queue import Queue, Empty
from mqtt_client import build_client
from infer import InferenceEngine
from config import EDGE_ID, HOUSE_ID

# Thread-safe queue
queue = Queue()
client = build_client(queue)
engine = InferenceEngine()

def mqtt_loop():
    client.loop_forever()

def processing_loop():
    while True:
        try:
            # wait up to 1 second for a message
            item = queue.get(timeout=1.0)
        except Empty:
            continue  # no message, loop again

        try:
            ct_samples = item.get("ct_sample", [])
            preds, latency = engine.infer_nilm(ct_samples)
            if preds:
                msg = {
                    "ts": item.get("ts"),
                    "edge_id": EDGE_ID,
                    "house_id": HOUSE_ID,
                    "model": "nilm-v1.tflite",
                    "inference_type": "nilm",
                    "predictions": preds,
                    "latency_ms": latency
                }
                client.publish(
                    f"home/{HOUSE_ID}/edge/{EDGE_ID}/inference",
                    json.dumps(msg)
                )
                print(f"[Inference Published] {msg}")
        except Exception as e:
            print(f"[Processing Error] {e}")
        finally:
            queue.task_done()  # mark message as processed

if __name__ == "__main__":
    # start MQTT listener
    threading.Thread(target=mqtt_loop, daemon=True).start()
    print("Edge inference service started...")
    processing_loop()
