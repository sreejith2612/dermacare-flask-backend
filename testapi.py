import requests

# URL of the Flask app
url = 'http://127.0.0.1:5000/predict'

# Image file path
image_path = r"C:\Users\sreej\Downloads\ISIC_0024306.jpg"

# Send POST request with the image file
with open(image_path, 'rb') as img:
    files = {'image': img}
    response = requests.post(url, files=files)

# Print the server's response
print(response.json())
