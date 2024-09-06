from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import httpx
import base64
import io
from PIL import Image
import json
import hashlib
import logging
import re
from google.cloud import translate


app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CONTROLLER_ADDRESS = "http://localhost:10000"  # Replace with your controller's address
WORKER_TIMEOUT = 60.0  # Increased timeout

# Initialize Translation client
def translate_text(
    text: str, project_id: str = "apt-market-430913-t8"
    ):

    client = translate.TranslationServiceClient()

    location = "global"

    parent = f"projects/{project_id}/locations/{location}"

    # Translate text from English to French
    # Detail on supported types can be found here:
    # https://cloud.google.com/translate/docs/supported-formats
    response = client.translate_text(
        request={
            "parent": parent,
            "contents": [text],
            "mime_type": "text/plain",  # mime types: text/plain, text/html
            "source_language_code": "en-US",
            "target_language_code": "id",
        }
    )

    # Display the translation for each input text provided
    for translation in response.translations:
        return translation.translated_text

async def get_worker_address(model: str) -> str:
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{CONTROLLER_ADDRESS}/get_worker_address", json={"model": model})
            response.raise_for_status()
            return response.json()["address"]
        except httpx.HTTPError as e:
            logger.error(f"Failed to get worker address: {str(e)}")
            raise HTTPException(status_code=503, detail="Failed to get worker address from controller")

async def process_image(image: UploadFile) -> tuple:
    image_content = await image.read()
    image = Image.open(io.BytesIO(image_content)).convert('RGB')
    image_hash = hashlib.md5(image.tobytes()).hexdigest()
    
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return image_hash, base64_image

async def stream_generator(worker_address: str, payload: dict):
    async with httpx.AsyncClient() as client:
        async with client.stream('POST', f"{worker_address}/worker_generate_stream", json=payload, timeout=WORKER_TIMEOUT) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    logger.debug(f"Received line from worker: {line}")
                    try:
                        data = json.loads(line)
                        if "error_code" in data:
                            if data["error_code"] == 0 and "text" in data:
                                yield f"data: {data['text']}\n\n"
                            else:
                                logger.error(f"Received error from worker: {data}")
                                yield f"data: Error: {data.get('text', 'Unknown error')}\n\n"
                        else:
                            logger.warning(f"Received JSON data without 'error_code' field: {data}")
                    except json.JSONDecodeError:
                        logger.warning(f"Received non-JSON line, treating as raw text: {line}")
                        yield f"data: {line.strip()}\n\n"
                    except Exception as e:
                        logger.error(f"Unexpected error processing line: {str(e)}")
            yield "data: [DONE]\n\n"

@app.get("/")
async def root():
    return {"health_check": "OK", "model_version": "v1.0"}

@app.post("/generate")
async def generate(image: UploadFile = File(...), prompt: str = Form(...)):
    try:
        image_hash, base64_image = await process_image(image)
        logger.info(f"Processed image. Hash: {image_hash}")
        
        model_name = "llava-med-v1.5-mistral-7b"
        worker_address = await get_worker_address(model_name)
        logger.info(f"Got worker address: {worker_address}")
        
        payload = {
            "model": model_name,
            "prompt": f"[INST] <image>\n{prompt} [/INST]",
            "temperature": 0.2,
            "top_p": 0.7,
            "max_new_tokens": 512,
            "stop": "</s>",
            "images": [base64_image]
        }
        logger.debug(f"Prepared payload: {payload}")

        async with httpx.AsyncClient() as client:
            logger.info(f"Sending request to worker at {worker_address}")
            response = await client.post(f"{worker_address}/worker_generate_stream", json=payload, timeout=WORKER_TIMEOUT)
            
            response.raise_for_status()

            chunks = []
            async for chunk in response.aiter_bytes():
                chunks.append(chunk)

            if not chunks:
                logger.warning("Received empty response from worker")
                raise HTTPException(status_code=500, detail="Worker returned empty response")

            # Process the last chunk
            last_chunk = chunks[-1]
            chunk_str = last_chunk.decode('utf-8')
            json_objects = chunk_str.split('\u0000')
            full_text = ""

            for json_str in reversed(json_objects):
                if json_str.strip():
                    try:
                        data = json.loads(json_str)
                        if "text" in data:
                            full_text = data["text"]
                            break  # Exit loop after finding the last valid JSON with 'text'
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse JSON: {json_str}")

            if not full_text:
                logger.warning("No valid 'text' field found in the response")
                raise HTTPException(status_code=500, detail="No valid text found in worker response")
            
            # Clean up the response
            generated_text = full_text.replace(f"[INST] <image>\n{prompt} [/INST]", "").strip()
            translated_text = translate_text(generated_text)
            
            logger.info(f"Final generated text: {generated_text}")
            
            return {"generated_text": translated_text}

    except httpx.RequestError as e:
        logger.error(f"Network error occurred: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unavailable due to network error")
    except httpx.TimeoutException:
        logger.error("Request to worker timed out")
        raise HTTPException(status_code=504, detail="Worker request timed out")
    except Exception as e:
        logger.exception("Unexpected error in generate endpoint")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)