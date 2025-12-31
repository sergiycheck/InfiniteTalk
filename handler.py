import multiprocessing as mp
import os
from generate_infinitetalk import generate
import json
from boto3_utils import download_s3_file, upload_s3_file
from utils import now_local_str
import uuid

def infinite_talk_worker(image_s3_key: str, audio_s3_key: str, text_prompt: str):
    try:
        
        bucket_name = os.getenv("S3_BUCKET_NAME")
        
        if not bucket_name:
            raise RuntimeError("S3_BUCKET_NAME environment variable is not set")
        
        output_path = os.path.join("output")
        
        os.makedirs(output_path, exist_ok=True)

        
        print("Downloding image from s3", now_local_str())
        
        image_path = download_s3_file(
            bucket=bucket_name,
            key=image_s3_key,
            local_path=output_path
        )
        
        print("Downloding audio from s3", now_local_str())
        
        audio_path = download_s3_file(
            bucket=bucket_name,
            key=audio_s3_key,
            local_path=output_path
        )
        
        generated_video_path = os.path.join(output_path, f"infinitetalk_{uuid.uuid4().hex}.mp4")

        input_json_content = {
            "prompt": text_prompt,
            "cond_video": image_path,
            "cond_audio": {
            "person1": audio_path
            }
        }
        
        print("Generating video...", now_local_str())
        
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
        
        upload_s3_file(
            local_path=generated_video_path,
            bucket=bucket_name,
        )
            
        generated_video_name = os.path.basename(generated_video_path)
        s3_url = f"https://{bucket_name}.s3.amazonaws.com/{generated_video_name}"
        
        print("Job completed.", now_local_str())
        print("Generated video S3 URL:", s3_url)
        
        return({
            "status": "completed",
            "s3_url": s3_url
        })

    except Exception as e:
        print(f"Error in infinite_talk_worker: {e}")
        
        return({
            "status": "error",
            "error": str(e)
        })

    finally:
        os.remove(image_path)
        os.remove(audio_path)
        os.remove(generated_video_path)

