import json
import math
from tqdm import tqdm

def generate_summary_with_progress(input_path, output_path):


    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)


    def bbox_center(bbox):
        x, y, w, h = bbox
        return (x + w/2, y + h/2)

    def euclidean_distance(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def classify_distance(distance):
        if distance < 50:
            return "extremely near"
        elif distance < 150:
            return "near"
        elif distance < 300:
            return "moderately near"
        elif distance < 600:
            return "moderately far"
        elif distance < 1000:
            return "far"
        else:
            return "extremely far"

    distance_categories_order = [
        "extremely near", "near", "moderately near",
        "moderately far", "far", "extremely far"
    ]
    frames_out = []

    for frame_key, detections in tqdm(data.items(),
                                      total=len(data),
                                      desc="Processing frames"):
        frame_id = int(frame_key.split("_")[-1])

        cars_list = []
        persons_list = []
        obj_info = {}

 
        for obj in detections:
            obj_id = f"id{obj['track_id']}" if "track_id" in obj else obj["label"]
            obj_info[obj_id] = {
                "center": bbox_center(obj["bbox"]),
                "type": obj["label"]
            }

            if obj["label"].lower() == "car":
                cars_list.append(f"{obj_id} {obj['color']}")
            elif obj["label"].lower() == "person":
          
                posture = obj.get("posture", "unknown posture")
                persons_list.append(f"{obj_id} color is {obj['color']} and posture is {posture}")

        total_cars = len(cars_list)
        total_persons = len(persons_list)

        part1 = "Cars: " + ", ".join(cars_list) + f". Total cars: {total_cars}"
        part2 = "Persons: " + ", ".join(persons_list) + f". Total persons: {total_persons}"


        distance_groups = {cat: [] for cat in distance_categories_order}
        keys = list(obj_info.keys())
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                id1, id2 = keys[i], keys[j]
                dist = euclidean_distance(obj_info[id1]["center"], obj_info[id2]["center"])
                cat = classify_distance(dist)
                distance_groups[cat].append(f"{id1}–{id2}")

   
        line_ext_near = "Extremely near: " + "; ".join(distance_groups["extremely near"] or ["None"])
        distance_line1 = f"{line_ext_near}" + ". "
        line_ext_near = "Extremely near: " + "; ".join(distance_groups["extremely near"] or ["None"])
        distance_line1 = f"{line_ext_near}" + ". "
      

        summary_str = (
            f"{part1}\n"
            f"{part2}\n"
            f"{distance_line1}"
        )

        frames_out.append({
            "frame_id": frame_id,
            "summary": summary_str
        })

    output_json = {"frames": frames_out}
    with open(output_path, "w", encoding="utf-8") as out_f:
        json.dump(output_json, out_f, indent=4)



generate_summary_with_progress("step_1.json", "step_2.json")
