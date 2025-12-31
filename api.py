from dotenv import load_dotenv
load_dotenv()

import os
import uuid
import json
import multiprocessing as mp
import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException

from pydantic import BaseModel
import sys
import os
import uuid
from boto3_utils import download_s3_file, upload_s3_file
from generate_infinitetalk import generate
import json

from boto3_utils import download_s3_file, upload_s3_file
from datetime import datetime

MAX_CONCURRENT_JOBS = 1
job_semaphore = mp.Semaphore(MAX_CONCURRENT_JOBS)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = FastAPI()
active_connections: Dict[str, WebSocket] = {}
queues: Dict[str, mp.Queue] = {}

def now_local_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

class VideoRequest(BaseModel):
    text_prompt: str
    image_s3_key: str
    audio_s3_key: str

app = FastAPI()

def infinite_talk_worker(job_id: str, request: dict, queue: mp.Queue):
    bucket_name = os.getenv("S3_BUCKET_NAME")
    
    if not bucket_name:
        raise RuntimeError("S3_BUCKET_NAME environment variable is not set")
      
    output_path = os.path.join("output")
    
    if(not os.path.exists("output")):
        os.makedirs("output")
        

    try:
    
        print("Downloding image from s3", now_local_str())
        queue.put({"status": "downloading_image"})
        
        image_path = download_s3_file(
            bucket=bucket_name,
            key=request.image_s3_key,
            local_path=output_path
        )
        
        print("Downloding audio from s3", now_local_str())
        queue.put({"status": "downloading_audio"})
        
        audio_path = download_s3_file(
            bucket=bucket_name,
            key=request.audio_s3_key,
            local_path=output_path
        )
        
        generated_video_path = os.path.join(output_path, f"infinitetalk_{uuid.uuid4().hex}.mp4")
    
        input_json_content = {
            "prompt": request.text_prompt,
            "cond_video": image_path,
            "cond_audio": {
            "person1": audio_path
            }
        }
        
        print("Generating video...", now_local_str())
        queue.put({"status": "generating_video"})
        
        generate(
            ckpt_dir="weights/Wan2.1-I2V-14B-480P",
            wav2vec_dir="weights/chinese-wav2vec2-base",
            infinitetalk_dir="weights/InfiniteTalk/single/infinitetalk.safetensors",
            input_json=json.dumps(input_json_content),
            size="infinitetalk-480",
            sample_steps="40",
            mode="streaming",
            motion_frame="9",
            save_file=generated_video_path,
        )
        
        print("Generated video:", generated_video_path)
        queue.put({"status": "uploading_to_s3"})
        
        upload_s3_file(
            local_path=generated_video_path,
            bucket=bucket_name,
        )
            
        print("Job completed.", now_local_str())
        
        generated_video_name = os.path.basename(generated_video_path)
        s3_url = f"https://{bucket_name}.s3.amazonaws.com/{generated_video_name}"
        queue.put({
            "status": "completed",
            "s3_url": s3_url
        })

    except Exception as e:
        queue.put({
            "status": "error",
            "error": str(e)
        })

    finally:
        try:
            os.remove(image_path)
            os.remove(audio_path)
            os.remove(generated_video_path)
            job_semaphore.release()
        except Exception:
            pass


async def ws_event_forwarder(job_id: str, queues: Dict[str, mp.Queue]):
    websocket = active_connections.get(job_id)
    if not websocket:
        return

    while True:
        msg = await asyncio.to_thread(queues[job_id].get)
        await websocket.send_text(json.dumps(msg))
        if msg["status"] in ("completed", "error"):
            break

@app.post("/generate-video")
async def generate_video(request: VideoRequest):

    acquired = job_semaphore.acquire(block=False)
    if not acquired:
        raise HTTPException(429, "Server busy")
    
    job_id = str(uuid.uuid4())
    queue = mp.Queue()
    queues[job_id] = queue

    process = mp.Process(
        target=infinite_talk_worker,
        args=(job_id, request.dict(), queue, job_semaphore)
    )
    process.start()
    asyncio.create_task(ws_event_forwarder(job_id, queues))

    return {
        "job_id": job_id,
        "ws_url": f"/ws/{job_id}"
    }


@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    await websocket.accept()
    active_connections[job_id] = websocket

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_connections.pop(job_id, None)
