import os
import sys
import base64
import requests
import subprocess
import cv2
import argparse

def extract_last_frame(video_path, output_image_path):
    print(f"Extracting the final frame from {video_path}...")
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
    ret, frame = cap.read()

    if ret:
        cv2.imwrite(output_image_path, frame)
        print("Frame extracted successfully!")
    else:
        print("Failed to extract the last frame.")
        sys.exit(1)
    cap.release()

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def main():
    parser = argparse.ArgumentParser(description="AgentFM 10-Second 4K Studio Director")
    parser.add_argument("prompt", type=str, help="The prompt with the '|' delimiter")
    parser.add_argument("--upscale", action="store_true", help="Send the stitched 10s video to Port 8002 for 4K Upscaling")
    args = parser.parse_args()

    url_i2v = "http://localhost:8001/api/generate"
    url_upscale = "http://localhost:8002/api/upscale"

    print("\nGenerating Part 1 (0s - 5s)")
    payload_1 = {"prompt": args.prompt}

    try:
        response_1 = requests.post(url_i2v, json=payload_1)
        response_1.raise_for_status()
        with open("part1.mp4", "wb") as f:
            f.write(response_1.content)
    except Exception as e:
        print(f"Part 1 failed: {e}")
        sys.exit(1)

    extract_last_frame("part1.mp4", "bridge_frame.jpg")
    encoded_frame = encode_image("bridge_frame.jpg")

    print("\nGenerating Part 2 (5s - 10s)...")
    payload_2 = {
        "prompt": args.prompt,
        "init_image_b64": encoded_frame 
    }

    try:
        response_2 = requests.post(url_i2v, json=payload_2)
        response_2.raise_for_status()
        with open("part2.mp4", "wb") as f:
            f.write(response_2.content)
    except Exception as e:
        print(f"Part 2 failed: {e}")
        sys.exit(1)

    print("\n Stitching the clips together into a 10-second movie...")
    with open("stitch_list.txt", "w") as f:
        f.write("file 'part1.mp4'\n")
        f.write("file 'part2.mp4'\n")

    raw_10s_file = 'final_10s_raw.mp4'
    try:
        subprocess.run([
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
            '-i', 'stitch_list.txt', '-c', 'copy', raw_10s_file
        ], check=True, capture_output=True)
        print(f"✅ 10-second raw movie saved as '{raw_10s_file}'")
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg stitching failed: {e.stderr.decode()}")
        sys.exit(1)

    if args.upscale:
        print(f"\n✨ Sending 10-second video to Post-Production (Port 8002) for 4K Upscaling...")
        try:
            with open(raw_10s_file, "rb") as f:
                video_binary = f.read()

            files = {"file": ("raw_10s.mp4", video_binary, "video/mp4")}
            upscale_response = requests.post(url_upscale, files=files)
            upscale_response.raise_for_status()

            final_4k_file = "final_10s_4K_MASTERPIECE.mp4"
            with open(final_4k_file, "wb") as f:
                f.write(upscale_response.content)

            print(f"🎉 SUCCESS! 10-Second 4K Masterpiece saved to: {final_4k_file}")

        except Exception as e:
            print(f"❌ Upscaling failed: {e}")
    else:
        print("🎉 SUCCESS! Render complete. (Run with --upscale for 4K)")

    for file in ["part1.mp4", "part2.mp4", "bridge_frame.jpg", "stitch_list.txt"]:
        if os.path.exists(file):
            os.remove(file)

if __name__ == "__main__":
    main()