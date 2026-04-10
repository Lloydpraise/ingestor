import logging
import os, json, base64, requests, subprocess, shutil, uuid, time
from flask import Flask, request, jsonify

# Set up logging early
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("ingestor")

app = Flask(__name__)

# --- LEAPCELL HEALTH CHECK ---
@app.route('/health')
def health_check():
    return "OK", 200

@app.route('/')
def root():
    return jsonify({"status": "alive"}), 200

# --- UTILS ---
def get_config():
    conf = {
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
        "SUPABASE_URL": os.environ.get("SUPABASE_URL"),
        "SUPABASE_KEY": os.environ.get("SUPABASE_KEY"),
        "DINO_ENDPOINT": os.environ.get("DINO_ENDPOINT")
    }
    return conf

# --- CORE LOGIC ---
def process_video(video_url):
    # HEAVY IMPORTS ONLY WHEN CALLED
    import cv2
    from openai import OpenAI
    from supabase import create_client

    logger.info("Starting ingest: %s", video_url)
    conf = get_config()
    client = OpenAI(api_key=conf["OPENAI_API_KEY"])
    supabase = create_client(conf["SUPABASE_URL"], conf["SUPABASE_KEY"])
    
    run_id = str(uuid.uuid4())[:8]
    video_path = f"/tmp/video_{run_id}.mp4"
    frames_dir = f"/tmp/frames_{run_id}"
    os.makedirs(frames_dir, exist_ok=True)

    try:
        # 1. Download
        subprocess.run(["yt-dlp", "--no-playlist", "-o", video_path, video_url], check=True, timeout=180)

        # 2. Extract
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        interval = max(1, int(fps * 0.5))
        
        frame_list = []
        count = 0
        while len(frame_list) < 10:
            ret, frame = cap.read()
            if not ret: break
            if count % interval == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if cv2.Laplacian(gray, cv2.CV_64F).var() > 50:
                    fpath = os.path.join(frames_dir, f"f_{count}.jpg")
                    cv2.imwrite(fpath, frame)
                    frame_list.append({"name": f"f_{count}.jpg", "path": fpath})
            count += 1
        cap.release()

        # 3. AI
        payload_imgs = []
        for f in frame_list[:6]:
            with open(f["path"], "rb") as im:
                b64 = base64.b64encode(im.read()).decode('utf-8')
                payload_imgs.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "Return JSON: {'caption_prefix': 'str', 'shoes': [{'frame_name': 'str', 'coords': [ymin, xmin, ymax, xmax]}]}"},
                *payload_imgs
            ]}]
        )
        raw_data = json.loads(response.choices[0].message.content)
        
        # 4. Crops & Upload
        processed_items = []
        for i, shoe in enumerate(raw_data.get('shoes', [])):
            target = next((f for f in frame_list if f["name"] == shoe['frame_name']), frame_list[0])
            img = cv2.imread(target["path"])
            h, w, _ = img.shape
            c = shoe['coords']
            y1, x1, y2, x2 = [int(c[0]*h/1000), int(c[1]*w/1000), int(c[2]*h/1000), int(c[3]*w/1000)]
            
            crop_path = f"/tmp/cp_{run_id}_{i}.jpg"
            cv2.imwrite(crop_path, img[y1:y2, x1:x2])

            s_path = f"ingests/{run_id}/shoe_{i}.jpg"
            with open(crop_path, 'rb') as f_up:
                supabase.storage.from_("shoe-crops").upload(s_path, f_up, {"content-type": "image/jpeg"})
            
            pub_url = supabase.storage.from_("shoe-crops").get_public_url(s_path)
            
            dino_payload = {
                "video_id": f"vid_{run_id}",
                "image_url": str(pub_url),
                "caption_prefix": raw_data.get('caption_prefix', 'New Arrival')
            }
            requests.post(f"{conf['DINO_ENDPOINT']}/index-video-frame", json=dino_payload, timeout=30)
            processed_items.append(dino_payload)

        return processed_items

    finally:
        if os.path.exists(video_path): os.remove(video_path)
        if os.path.exists(frames_dir): shutil.rmtree(frames_dir)

@app.route('/ingest', methods=['POST'])
def ingest():
    data = request.get_json()
    try:
        res = process_video(data['url'])
        return jsonify({"status": "success", "data": res})
    except Exception as e:
        logger.exception("Ingest failed")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)