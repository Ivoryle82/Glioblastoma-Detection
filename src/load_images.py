import requests
from PIL import Image
from io import BytesIO

# URL of the image you want to download
image_url = 'https://springernature.figshare.com/ndownloader/files/38249697'

# Send a GET request to the URL to download the image
response = requests.get(image_url)

# Check if the request was successful
if response.status_code == 200:
    # Read the content of the response
    image_data = response.content
    
    # Create a PIL image object from the image data
    image = Image.open(BytesIO(image_data))
    
    # Display or manipulate the image as needed
    image.show()
else:
    print("Failed to download image:", response.status_code)
