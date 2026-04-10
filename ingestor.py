import logging
import os, cv2, json, base64, requests, subprocess, shutil, uuid, time
from flask import Flask, request, jsonify
from openai import OpenAI
from supabase import create_client, Client

app = Flask(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("ingestor")

# --- UTILS ---
def get_config():
    conf = {
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
        "SUPABASE_URL": os.environ.get("SUPABASE_URL"),
        "SUPABASE_KEY": os.environ.get("SUPABASE_KEY"),
        "DINO_ENDPOINT": os.environ.get("DINO_ENDPOINT")
    }
    missing = [k for k, v in conf.items() if not v]
    if missing:
        logger.error("Missing required env vars: %s", missing)
        raise Exception(f"Missing Env Vars: {missing}")
    return conf

def safe_int(val, default=0):
    try: return int(val)
    except: return default

# --- CORE LOGIC ---
def process_video(video_url):
    logger.info("Starting ingest for URL: %s", video_url)
    conf = get_config()
    client = OpenAI(api_key=conf["OPENAI_API_KEY"])
    supabase: Client = create_client(conf["SUPABASE_URL"], conf["SUPABASE_KEY"])
    
    run_id = str(uuid.uuid4())[:8]
    
    # CRITICAL: Use /tmp/ for all file operations on Leapcell
    video_path = f"/tmp/video_{run_id}.mp4"
    frames_dir = f"/tmp/frames_{run_id}"
    os.makedirs(frames_dir, exist_ok=True)

    try:
        # 1. DOWNLOAD (with retry)
        logger.info("Downloading video to %s", video_path)
        try:
            subprocess.run(["yt-dlp", "--no-playlist", "-o", video_path, video_url], check=True, timeout=180)
            logger.info("Download completed for %s", video_url)
        except Exception as e:
            logger.error("Video download failed for %s: %s", video_url, e)
            raise Exception(f"Video download failed: {str(e)}")

        # 2. EXTRACTION WITH VALIDATION
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Could not open video file: %s", video_path)
            raise Exception("Could not open video file.")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            logger.warning("FPS detection failed (%s). Defaulting to 30", fps)
            fps = 30 
        interval = max(1, int(fps * 0.5))
        logger.info("Video opened. fps=%s interval=%s", fps, interval)
        
        frame_list = []
        count = 0
        while len(frame_list) < 10: # Cap extraction to 10 frames
            ret, frame = cap.read()
            if not ret: break
            if count % interval == 0:
                # Sharpness check (Laplacian)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                if sharpness > 50:
                    fname = f"frame_{count}.jpg"
                    fpath = os.path.join(frames_dir, fname)
                    cv2.imwrite(fpath, frame)
                    frame_list.append({"name": fname, "path": fpath})
                    logger.debug("Saved sharp frame %s (sharpness=%s)", fname, sharpness)
                else:
                    logger.debug("Skipped frame %s due to low sharpness=%s", count, sharpness)
            count += 1
        cap.release()

        logger.info("Extracted %s usable frames", len(frame_list))
        if not frame_list:
            logger.error("No usable frames extracted from video %s", video_url)
            raise Exception("No usable frames extracted from video.")

        # 3. AI VISION WITH STRUCTURED OUTPUT
        payload_imgs = []
        for f in frame_list[:6]:
            with open(f["path"], "rb") as im_file:
                b64 = base64.b64encode(im_file.read()).decode('utf-8')
                payload_imgs.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        logger.info("Prepared %s frames for OpenAI request", len(payload_imgs))

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "Return JSON: {'caption_prefix': 'str', 'shoes': [{'frame_name': 'str', 'coords': [ymin, xmin, ymax, xmax]}]}. Coords must be 0-1000 scale."},
                *payload_imgs
            ]}]
        )
        logger.info("OpenAI response received")
        raw_data = json.loads(response.choices[0].message.content)
        
        if 'shoes' not in raw_data or not isinstance(raw_data['shoes'], list):
            logger.error("Invalid AI response format: %s", raw_data)
            raise Exception("Invalid AI response format.")

        # 4. PROCESSING RESULTS
        processed_items = []
        logger.info("Processing %s detected shoes", len(raw_data['shoes']))
        for i, shoe in enumerate(raw_data['shoes']):
            if not all(k in shoe for k in ('frame_name', 'coords')):
                logger.warning("Skipping shoe entry missing required keys: %s", shoe)
                continue
            
            target = next((f for f in frame_list if f["name"] == shoe['frame_name']), None)
            if target is None:
                logger.warning("Frame name not found: %s. Using first available.", shoe['frame_name'])
                target = frame_list[0]

            img = cv2.imread(target["path"])
            if img is None:
                logger.error("Failed to load image for frame %s", target["name"])
                continue

            h, w, _ = img.shape
            coords = shoe['coords']
            if not isinstance(coords, list) or len(coords) != 4:
                continue
            
            # Normalize and scale coords
            y1 = max(0, min(h, int(coords[0] * h / 1000)))
            x1 = max(0, min(w, int(coords[1] * w / 1000)))
            y2 = max(0, min(h, int(coords[2] * h / 1000)))
            x2 = max(0, min(w, int(coords[3] * w / 1000)))

            if y2 <= y1 or x2 <= x1:
                continue 

            # Save crop to /tmp/
            crop_path = f"/tmp/crop_{run_id}_{i}.jpg"
            cv2.imwrite(crop_path, img[y1:y2, x1:x2])
            logger.info("Created crop %s", crop_path)

            # STORAGE UPLOAD
            storage_path = f"ingests/{run_id}/shoe_{i}.jpg"
            try:
                with open(crop_path, 'rb') as f_upload:
                    supabase.storage.from_("shoe-crops").upload(storage_path, f_upload, {"content-type": "image/jpeg"})
                logger.info("Uploaded crop to Supabase path=%s", storage_path)
            except Exception as e:
                logger.error("Supabase upload failed: %s", e)
                continue
            
            # SUPABASE URL RESOLUTION
            res = supabase.storage.from_("shoe-crops").get_public_url(storage_path)
            pub_url = res.public_url if hasattr(res, 'public_url') else (res.get('publicUrl') if isinstance(res, dict) else res)

            if not pub_url:
                logger.error("Could not resolve public URL from response: %s", res)
                continue

            # DINO PING
            dino_payload = {
                "video_id": f"vid_{run_id}",
                "product_id": f"prod_{run_id}_{i}",
                "image_url": pub_url,
                "caption_prefix": raw_data.get('caption_prefix', 'New Arrival')
            }
            logger.info("Sending DINO request for product_id=%s", dino_payload['product_id'])
            try:
                r = requests.post(f"{conf['DINO_ENDPOINT']}/index-video-frame", json=dino_payload, timeout=45)
                r.raise_for_status()
                logger.info("DINO indexing succeeded")
                processed_items.append(dino_payload)
            except Exception as e:
                logger.error("Failed to index item %s: %s", i, e)

        return processed_items

    finally:
        logger.info("Cleaning up /tmp/ files for run_id=%s", run_id)
        if os.path.exists(video_path): os.remove(video_path)
        if os.path.exists(frames_dir): shutil.rmtree(frames_dir)
        # Cleanup crops specifically for this run
        for f in os.listdir('/tmp/'):
            if f.startswith(f"crop_{run_id}"): 
                try: os.remove(os.path.join('/tmp/', f))
                except: pass

@app.route('/kaithheathcheck')
def health_check():
    return "OK", 200

@app.route('/ingest', methods=['POST'])
def ingest():
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({"error": "No URL provided"}), 400
    
    url = data['url']
    logger.info("Received ingest request for URL: %s", url)
    try:
        results = process_video(url)
        return jsonify({"status": "success", "count": len(results), "data": results})
    except Exception as e:
        logger.exception("Error processing video URL %s", url)
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))