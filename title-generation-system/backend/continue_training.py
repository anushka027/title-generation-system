# continue_training.py
import requests
import sys
import json

if len(sys.argv) < 2:
    print("Usage: python continue_training.py <csv_path> [epochs] [learning_rate]")
    sys.exit(1)

csv_path = sys.argv[1]
epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 3
learning_rate = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-5  # Lower learning rate for continued training

print(f"Continuing training on {csv_path} for {epochs} epochs with learning rate {learning_rate}")

response = requests.post(
    'http://localhost:5000/train',
    json={
        'csv_path': csv_path,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'continue_training': True  # Flag to indicate continuing training
    }
)

print(response.json())