from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
import base64
import io
from PIL import Image
import json
import hashlib
from typing import List
import logging

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

            full_response = []
            async for line in response.aiter_lines():
                if line:
                    logger.debug(f"Received line from worker: {line}")
                    try:
                        # First, try to parse as JSON
                        data = json.loads(line)
                        if "error_code" in data:
                            if data["error_code"] == 0:
                                if "text" in data:
                                    full_response.append(data["text"].strip())
                                    logger.info(f"Appended response chunk: {data['text'].strip()}")
                                else:
                                    logger.warning(f"Received JSON data without 'text' field: {data}")
                            else:
                                logger.error(f"Received error from worker: {data}")
                        else:
                            logger.warning(f"Received JSON data without 'error_code' field: {data}")
                    except json.JSONDecodeError:
                        # If it's not JSON, treat the entire line as text
                        logger.warning(f"Received non-JSON line, treating as raw text: {line}")
                        full_response.append(line.strip())
                    except Exception as e:
                        logger.error(f"Unexpected error processing line: {str(e)}")
                    finally:
                        logger.debug("Continuing to next line")

            complete_response = "".join(full_response)
            
            if not complete_response:
                logger.warning("Received empty response from worker")
                raise HTTPException(status_code=500, detail="Worker returned empty response")
            
                        # Remove the trailing error code and quotation marks
            if complete_response.endswith('", "error_code": 0}'):
                generated_text = complete_response[:-len('", "error_code": 0}')]
            else:
                generated_text = complete_response

            # Remove any remaining trailing quotation marks
            generated_text = generated_text.rstrip('"')

            logger.info(f"Final generated text: {generated_text}")
            
            # Extract the generated text by removing the instruction
            instruction_end = complete_response.rfind("[/INST]")
            if instruction_end != -1:
                generated_text = complete_response[instruction_end + 7:].strip()
            else:
                generated_text = complete_response.strip()
            
            logger.info(f"Final generated text: {generated_text}")
            
            return {"generated_text": generated_text}
    
    except httpx.RequestError as e:
        logger.error(f"Network error occurred: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unavailable due to network error")
    except httpx.TimeoutException:
        logger.error("Request to worker timed out")
        raise HTTPException(status_code=504, detail="Worker request timed out")
    except Exception as e:
        logger.exception("Unexpected error in generate endpoint")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.post("/chat")
async def chat(messages: List[dict], images: List[UploadFile] = File(None)):
    try:
        model_name = "llava-med-v1.5-mistral-7b"
        worker_address = await get_worker_address(model_name)
        
        processed_images = []
        if images:
            for image in images:
                _, base64_image = await process_image(image)
                processed_images.append(base64_image)
        
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": 0.2,
            "top_p": 0.7,
            "max_new_tokens": 512,
            "stop": "</s>",
            "images": processed_images
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(f"{worker_address}/worker_generate_stream", json=payload, timeout=WORKER_TIMEOUT)
            
            response.raise_for_status()

            full_response = []
            async for line in response.aiter_lines():
                if line:
                    logger.debug(f"Received line from worker: {line}")
                    try:
                        # First, try to parse as JSON
                        data = json.loads(line)
                        if "error_code" in data:
                            if data["error_code"] == 0:
                                if "text" in data:
                                    full_response.append(data["text"].strip())
                                    logger.info(f"Appended response chunk: {data['text'].strip()}")
                                else:
                                    logger.warning(f"Received JSON data without 'text' field: {data}")
                            else:
                                logger.error(f"Received error from worker: {data}")
                        else:
                            logger.warning(f"Received JSON data without 'error_code' field: {data}")
                    except json.JSONDecodeError:
                        # If it's not JSON, treat the entire line as text
                        logger.warning(f"Received non-JSON line, treating as raw text: {line}")
                        full_response.append(line.strip())
                    except Exception as e:
                        logger.error(f"Unexpected error processing line: {str(e)}")
                    finally:
                        logger.debug("Continuing to next line")

            complete_response = "".join(full_response)
            
            if not complete_response:
                logger.warning("Received empty response from worker")
                raise HTTPException(status_code=500, detail="Worker returned empty response")

            return {"generated_text": complete_response}

    except httpx.RequestError as e:
        logger.error(f"Network error occurred: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unavailable due to network error")
    except httpx.TimeoutException:
        logger.error("Request to worker timed out")
        raise HTTPException(status_code=504, detail="Worker request timed out")
    except Exception as e:
        logger.exception("Unexpected error in chat endpoint")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)