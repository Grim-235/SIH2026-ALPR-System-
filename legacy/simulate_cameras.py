#!/usr/bin/env python
"""Generate synthetic ANPR detection data for demo purposes."""
import argparse
import random
import json
from datetime import datetime, timedelta
from alpr.database import (
    init_db, load_cameras_from_json, load_blacklist_from_file,
    insert_detection, check_blacklist, insert_alert
)

def generate_plate():
    states = ["MH", "DL", "KA", "TN", "AP", "UP", "GJ", "RJ"]
    state = random.choice(states)
    district = f"{random.randint(1, 99):02d}"
    letters = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=random.randint(1, 2)))
    number = f"{random.randint(1, 9999):04d}"
    return f"{state}{district}{letters}{number}"

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic ANPR data.")
    parser.add_argument("--cameras", default="cameras.json", help="path to cameras.json")
    parser.add_argument("--db", default="data/alpr.db", help="path to SQLite DB")
    parser.add_argument("--blacklist", default="blacklist.txt", help="path to blacklist file")
    parser.add_argument("--hours", type=float, default=4.0, help="hours of data to generate")
    parser.add_argument("--count", type=int, default=150, help="approximate number of detections")
    parser.add_argument("--clear", action="store_true", help="clear existing data")
    args = parser.parse_args()

    conn = init_db(args.db)
    
    if args.clear:
        conn.execute("DELETE FROM detections")
        conn.execute("DELETE FROM alerts")
        conn.commit()
        print("Cleared existing data.")

    load_cameras_from_json(conn, args.cameras)
    load_blacklist_from_file(conn, args.blacklist)

    # Load cameras to get IDs
    try:
        with open(args.cameras, 'r') as f:
            cameras_data = json.load(f)
        camera_ids = [cam["camera_id"] for cam in cameras_data]
    except Exception as e:
        print(f"Failed to load cameras: {e}")
        return
    
    if not camera_ids:
        print("No cameras found in cameras.json")
        return

    # Load blacklist to get some blacklisted plates
    blacklisted_plates = []
    try:
        with open(args.blacklist, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                blacklisted_plates.append(line)
    except FileNotFoundError:
        pass

    pool_size = max(20, args.count // 5)
    plates = []
    
    # Add some blacklisted plates
    for _ in range(min(3, len(blacklisted_plates))):
        if blacklisted_plates:
            plates.append(random.choice(blacklisted_plates))
            
    # Add random plates
    while len(plates) < pool_size:
        plates.append(generate_plate())

    now = datetime.now()
    start_time = now - timedelta(hours=args.hours)
    
    # We want more traffic during rush hours (8-10am and 5-7pm)
    def get_random_time():
        while True:
            t = start_time + timedelta(seconds=random.randint(0, int(args.hours * 3600)))
            hour = t.hour
            # Simple weight: higher during 8-10 and 17-19
            if 8 <= hour <= 10 or 17 <= hour <= 19:
                prob = 1.0
            else:
                prob = 0.3
            if random.random() <= prob:
                return t

    total_detections = 0
    total_alerts = 0
    
    for _ in range(args.count):
        plate = random.choice(plates)
        
        # Decide how many cameras this plate is seen at
        num_cams = random.choices([1, 2, len(camera_ids)], weights=[0.7, 0.2, 0.1])[0]
        num_cams = min(num_cams, len(camera_ids))
        
        selected_cams = random.sample(camera_ids, num_cams)
        
        base_time = get_random_time()
        
        for i, cam_id in enumerate(selected_cams):
            if i > 0:
                # Add a realistic gap (5 to 30 mins)
                base_time += timedelta(minutes=random.randint(5, 30))
                
                # If we passed 'now', stop
                if base_time > now:
                    break
                    
            det_conf = random.uniform(0.65, 0.95)
            ocr_conf = random.uniform(0.55, 0.90)
            bbox = (
                random.randint(100, 500),
                random.randint(100, 500),
                random.randint(550, 900),
                random.randint(550, 900)
            )
            track_id = random.randint(1000, 9999)
            frame_num = random.randint(1, 300)
            
            insert_detection(
                conn, plate, cam_id, base_time.isoformat(),
                det_conf, ocr_conf, bbox, track_id, frame_num
            )
            total_detections += 1
            
            reason = check_blacklist(conn, plate)
            if reason:
                insert_alert(conn, plate, cam_id, base_time.isoformat(), reason)
                total_alerts += 1
                
    conn.commit()
    conn.close()
    
    print(f"Generated {total_detections} detections and {total_alerts} alerts across {len(camera_ids)} cameras.")

if __name__ == "__main__":
    main()
