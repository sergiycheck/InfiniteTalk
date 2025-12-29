from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os
import uuid
from boto3_utils import download_s3_file, upload_s3_file
from generate_infinitetalk import generate
import json


class VideoRequest(BaseModel):
    text_prompt: str
    image_s3_key: str
    audio_s3_key: str

app = FastAPI()

@app.post("/generate-video")
def read_root(request: VideoRequest):
  
    bucket_name = os.getenv("S3_BUCKET_NAME")
    
    if not bucket_name:
        raise RuntimeError("S3_BUCKET_NAME environment variable is not set")
      
    output_path = os.path.join("output")
    
    if(not os.path.exists("output")):
        os.makedirs("output")
    
    image_path = download_s3_file(
        bucket=bucket_name,
        key=request.image_s3_key,
        local_path=output_path
    )
    
    audio_path = download_s3_file(
        bucket=bucket_name,
        key=request.audio_s3_key,
        local_path=output_path
    )
    
    generated_video_name = f"infinitetalk_{uuid.uuid4().hex}.mp4"
  
    input_json_content = {
        "prompt": request.text_prompt,
        "cond_video": image_path,
        "cond_audio": {
        "person1": audio_path
        }
    }
    
    generate(
        ckpt_dir="weights/Wan2.1-I2V-14B-480P",
        wav2vec_dir="weights/chinese-wav2vec2-base",
        infinitetalk_dir="weights/InfiniteTalk/single/infinitetalk.safetensors",
        input_json=json.dumps(input_json_content),
        size="infinitetalk-480",
        sample_steps="40",
        mode="streaming",
        motion_frame="9",
        save_file=generated_video_name,
    )
    
    print("Generated video:", generated_video_name)
    
    upload_s3_file(
        bucket=bucket_name,
        key=f"{generated_video_name}",
        local_path=generated_video_name,
    )
    
    # Clean up local files
    os.remove(image_path)
    os.remove(audio_path)
    os.remove(generated_video_name)
    
    full_generated_video_url = f"https://{bucket_name}.s3.amazonaws.com/{generated_video_name}"
    
    return {"video_url": full_generated_video_url}