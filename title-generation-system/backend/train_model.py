import requests
import sys
import json
import os

if len(sys.argv) < 2:
    print("Usage: python train_model.py <csv_path> [epochs] [learning_rate]")
    sys.exit(1)

csv_path = sys.argv[1]
epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 3
learning_rate = float(sys.argv[3]) if len(sys.argv) > 3 else 3e-5

# Use environment variable or config file to store the API URL
# This allows you to switch between local development and production
API_URL = os.environ.get("MODEL_API_URL", "http://localhost:5000")

response = requests.post(
    f'{API_URL}/train',
    json={
        'csv_path': csv_path,
        'epochs': epochs,
        'learning_rate': learning_rate
    }
)

print(response.json())
# import requests
# import sys
# import json

# if len(sys.argv) < 2:
#     print("Usage: python train_model.py <csv_path> [epochs] [learning_rate]")
#     sys.exit(1)

# csv_path = sys.argv[1]
# epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 3
# learning_rate = float(sys.argv[3]) if len(sys.argv) > 3 else 3e-5

# response = requests.post(
#     'http://localhost:5000/train',
#     json={
#         'csv_path': csv_path,
#         'epochs': epochs,
#         'learning_rate': learning_rate
#     }
# )

# print(response.json())