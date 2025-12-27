import requests
from PIL import Image
import io
import matplotlib.pyplot as plt

# url = "http://localhost:8000/predict"

url = 'http://k8s-default-oilspill-9babf0751b-292ba6ab37ce3e10.elb.us-east-1.amazonaws.com:9696/predict'

request = {
    "image_url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQowQT72b_WbwMYsMACan4JLU64nWhVv0CxtA&s"
}

# Send request
response = requests.post(url, json=request)

# Check success
response.raise_for_status()

# Convert bytes → image
img = Image.open(io.BytesIO(response.content))

# Display
plt.imshow(img)
plt.axis("off")
plt.show()
