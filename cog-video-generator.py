import os
import io
import uuid
import logging
import torch
import requests
import threading
import subprocess
import base64
from fastapi import FastAPI, Response, HTTPException
from pydantic import BaseModel
from PIL import Image

from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video

class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage().find("GET /api/progress") == -1

logging.getLogger("uvicorn.access").addFilter(EndpointFilter())

app = FastAPI(title="CogVideoX Cinematic Wrapper")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
print("🟢 Initializing the massive CogVideoX-5B-I2V Engine...")

pipe = CogVideoXImageToVideoPipeline.from_pretrained(
    "THUDM/CogVideoX-5b-I2V",
    torch_dtype=torch.bfloat16
)

pipe.enable_model_cpu_offload(gpu_id=1)

try:
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    print("✅ 3D VAE Tiling & Slicing activated!")
except Exception as e:
    print(f"⚠️ ERROR enabling 3D tiling: {e}")

print("CogVideoX is locked on GPU 1!")

class SingleRequest(BaseModel):
    prompt: str
    init_image_b64: str = None

progress_state = {"status": "Idle"}
gpu_lock = threading.Lock()

@app.get("/api/progress")
async def get_progress():
    return progress_state

@app.post("/api/generate")
def generate_video(req: SingleRequest):
    with gpu_lock:
        if req.prompt.count("|") != 1:
            raise HTTPException(status_code=400, detail="Use one '|' to separate Image/Motion prompts.")

        image_prompt, motion_prompt = req.prompt.split("|")
        image_prompt, motion_prompt = image_prompt.strip(), motion_prompt.strip()

        image_words = image_prompt.split()
        core_subject = " ".join(image_words[:30])
        combined_video_prompt = f"{core_subject}... {motion_prompt}"

        print(f"\n🎬 NEW STUDIO JOB STARTING")
        print(f"Smart Prompt --->: '{combined_video_prompt}'")

        if req.init_image_b64:
            progress_state["status"] = "Injecting Frame..."
            image_data = base64.b64decode(req.init_image_b64)
            #  720x480
            init_image = Image.open(io.BytesIO(image_data)).convert("RGB").resize((720, 480))
        else:
            progress_state["status"] = "Generating FLUX Anchor..."
            try:
                flux_response = requests.post("http://localhost:8000/api/generate", json={"prompt": image_prompt})
                flux_response.raise_for_status()
                init_image = Image.open(io.BytesIO(flux_response.content)).convert("RGB").resize((720, 480))
            except Exception as e:
                progress_state["status"] = "Idle"
                raise HTTPException(status_code=500, detail=f"FLUX Error: {e}")

        def progress_callback(pipeline, step_index, timestep, callback_kwargs):
            progress_state["status"] = f"Animating Frame {step_index + 1} of 50..."
            return callback_kwargs

        try:
            print("🎥 Rendering CogVideoX Cinematic Motion...")

            video_frames = pipe(
                image=init_image,
                prompt=combined_video_prompt,
                num_inference_steps=50,
                guidance_scale=6.0,
                callback_on_step_end=progress_callback
            ).frames[0]

            raw_temp = f"/tmp/raw_{uuid.uuid4().hex[:8]}.mp4"
            final_temp = f"/tmp/hd_{uuid.uuid4().hex[:8]}.mp4"

            # CogVideoX outputs at 8fps naturally
            export_to_video(video_frames, raw_temp, fps=8)

            try:
                subprocess.run([
                    'ffmpeg', '-y', '-i', raw_temp,
                    '-c:v', 'libx264', '-crf', '17',
                    '-pix_fmt', 'yuv420p', final_temp
                ], check=True, capture_output=True)
                target_file = final_temp
            except Exception as e:
                target_file = raw_temp

            with open(target_file, "rb") as f:
                video_binary = f.read()

            for f in [raw_temp, final_temp]:
                if os.path.exists(f): os.remove(f)

            print(f"✅ Finished! Output: {len(video_binary)/1024/1024:.2f} MB")
            return Response(content=video_binary, media_type="video/mp4")

        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            progress_state["status"] = "Idle"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)