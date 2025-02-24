import base64
import io
import time

from openai import OpenAI
from PIL import Image

image = Image.open("path to image")

byte_stream = io.BytesIO()
image.save(byte_stream, format='JPEG')
byte_data = byte_stream.getvalue()
image_string = base64.b64encode(byte_data).decode('utf-8')

openai_api_key = "EMPTY"
openai_api_base = "http://127.0.0.1:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

start = time.time()
for i in range(10):
    chat_response = client.chat.completions.create(
        model="path to local path of llava-1.5-7b-hf",
        messages=[{
            "role": "user",
            "content": [
                # NOTE: The prompt formatting with the image token `<image>` is not needed
                # since the prompt will be processed automatically by the API server.
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_string}",
                    },
                },
            ],
        }],
    )
    print("Chat response:", chat_response)

end = time.time()
print(end - start)
