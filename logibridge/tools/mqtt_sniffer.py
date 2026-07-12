"""Quick MQTT sniffer for debugging pipeline flow."""
import sys
import time
import paho.mqtt.client as mqtt

TRUCK_ID = "TRUCK_001"
TOPICS = [
    f"logibridge/trucks/{TRUCK_ID}/sensors/temperature",
    f"logibridge/trucks/{TRUCK_ID}/sensors/vibration_rms",
    f"logibridge/trucks/{TRUCK_ID}/sensors/door_event",
    f"logibridge/trucks/{TRUCK_ID}/inference",
    f"logibridge/trucks/{TRUCK_ID}/alerts",
]

counts = {t: 0 for t in TOPICS}

def on_connect(client, _u, _f, rc):
    print(f"[SNIFF] connected rc={rc}")
    for t in TOPICS:
        client.subscribe(t, qos=1)
    print(f"[SNIFF] subscribed to {len(TOPICS)} topics")

def on_message(_c, _u, msg):
    counts[msg.topic] = counts.get(msg.topic, 0) + 1
    if counts[msg.topic] <= 2:
        print(f"[SNIFF] {msg.topic}: {msg.payload.decode('utf-8')[:150]}")

client = mqtt.Client(client_id="logibridge-sniffer")
client.on_connect = on_connect
client.on_message = on_message
client.connect("localhost", 1883, keepalive=60)
client.loop_start()

duration = int(sys.argv[1]) if len(sys.argv) > 1 else 15
print(f"[SNIFF] listening for {duration}s ...")
time.sleep(duration)
client.loop_stop()
client.disconnect()

print("\n[SNIFF] === MESSAGE COUNTS ===")
for t, c in counts.items():
    print(f"  {c:4d}  {t}")
