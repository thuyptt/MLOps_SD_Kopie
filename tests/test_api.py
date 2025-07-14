import requests

# Define the API endpoint
url = "https://cv-imagegen-test-tnesdf2hpa-ew.a.run.app/generate-image-file/"

# Define the request for the form data
request = {
    'prompt': 'Professional portrait for a CV. Blonde woman should appear in business outfit, with a white background suitable for professional settings. complete upper body. detailed, 8k',
    'image_url': 'https://i.ibb.co/VtkdCq8/michael-dam-m-EZ3-Po-FGs-k-unsplash-1.jpg'
}

# Send the POST request to the API endpoint
response = requests.post(url, data=request)

# Check if the request was successful
if response.status_code == 200:
    # Save the response content to a file
    with open('generated_image.jpg', 'wb') as f:
        f.write(response.content)
    print("Image saved as generated_image.jpg")
else:
    print(f"Failed to generate image. Status code: {response.status_code}")
    print(response.text)
