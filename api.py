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
    image_s3_link: str
    audio_s3_link: str

app = FastAPI()

@app.post("/generate-video")
def read_root(request: VideoRequest):
  
    bucket_name = "video_generation_bucket"
    bucket_folder = "generated_videos"
  
    output_path = os.path.join("output", f"{uuid.uuid4()}.wav")
    
    image_path = download_s3_file(
        bucket=bucket_name,
        key=request.image_s3_link.replace(f"s3://{bucket_name}/", ""),
        local_path=os.path.join("temp", f"{uuid.uuid4()}_ref.wav")
    )
    
    audio_path = download_s3_file(
        bucket=bucket_name,
        key=request.audio_s3_link.replace(f"s3://{bucket_name}/", ""),
        local_path=os.path.join("temp", f"{uuid.uuid4()}_ref.wav")
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
        key=f"{bucket_folder}/{generated_video_name}",
        local_path=generated_video_name,
    )
    
    # Clean up local files
    os.remove(image_path)
    os.remove(audio_path)
    os.remove(generated_video_name)
    
    full_generated_video_url = f"https://{bucket_name}.s3.amazonaws.com/{bucket_folder}/{generated_video_name}"
    
    return {"video_url": full_generated_video_url}