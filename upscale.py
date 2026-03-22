import sys
import torchvision.transforms.functional
sys.modules['torchvision.transforms.functional_tensor'] = torchvision.transforms.functional

import os
import cv2
import uuid
import torch
import subprocess
import shutil
from fastapi import FastAPI, UploadFile, File, Response, HTTPException
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

app = FastAPI(title="Real-ESRGAN 4K Post-Production")
print("Initializing Real-ESRGAN ")

model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

upsampler = RealESRGANer(
    scale=4,
    model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
    dni_weight=None,
    model=model,
    tile=400,
    tile_pad=10,
    pre_pad=0,
    half=True,
    device=torch.device('cuda:2')
)

print("Real-ESRGAN is locked on GPU 2!")

@app.post("/api/upscale")
async def upscale_video(file: UploadFile = File(...)):
    print(f"\n🎬  Upscaling to 4K...")

    session_id = uuid.uuid4().hex[:8]
    input_mp4 = f"/tmp/input_{session_id}.mp4"
    smooth_mp4 = f"/tmp/smooth_{session_id}.mp4"
    output_mp4 = f"/tmp/output_4k_{session_id}.mp4"
    frame_dir = f"/tmp/frames_{session_id}"

    os.makedirs(frame_dir, exist_ok=True)

    try:
        with open(input_mp4, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print("🪄 Generating missing frames to mooth 24 FPS...")
        subprocess.run([
            'ffmpeg', '-y', '-i', input_mp4,
            '-filter:v', 'minterpolate=mi_mode=mci:mc_mode=aobmc:vsbmc=1:fps=24',
            '-c:v', 'libx264', '-crf', '17', '-pix_fmt', 'yuv420p',
            smooth_mp4
        ], check=True, capture_output=True)

        cap = cv2.VideoCapture(smooth_mp4)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            upscaled_frame, _ = upsampler.enhance(frame, outscale=4)

            # Save the 4K frame
            frame_path = os.path.join(frame_dir, f"frame_{frame_idx:04d}.png")
            cv2.imwrite(frame_path, upscaled_frame)

            frame_idx += 1
            if frame_idx % 24 == 0:
                print(f"✨ Upscaled {frame_idx}/{total_frames} frames...")

        cap.release()
        print("🎥 Stitching  frames back into a  MP4...")

        subprocess.run([
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', f'{frame_dir}/frame_%04d.png',
            '-c:v', 'libx264',
            '-crf', '17',
            '-pix_fmt', 'yuv420p',
            output_mp4
        ], check=True, capture_output=True)

        with open(output_mp4, "rb") as f:
            video_binary = f.read()

        print(f"✅ Final Size: {len(video_binary)/1024/1024:.2f} MB")
        return Response(content=video_binary, media_type="video/mp4")

    except Exception as e:
        print(f"❌ ERROR during upscaling: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(input_mp4): os.remove(input_mp4)
        if os.path.exists(smooth_mp4): os.remove(smooth_mp4)
        if os.path.exists(output_mp4): os.remove(output_mp4)
        if os.path.exists(frame_dir): shutil.rmtree(frame_dir)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)