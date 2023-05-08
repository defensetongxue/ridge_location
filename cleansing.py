import json
import os

def parse_json(input_data):
    annotations = input_data.get("annotations", [])
    if annotations:
        result = annotations[0].get("result", [])

    new_data = {
        "image_name": input_data["file_upload"].split('-')[-1],
        "ridge_number": 0,
        "ridge_coordinate": [],
        "other_number": 0,
        "other_coordinate": [],
        "plus_number": 0,
        "plus_coordinate": [],
        "pre_plus_number": 0,
        "pre_plus_coordinate": []
    }

    for item in result:
        if item["type"] == "keypointlabels":
            # x, y = item["value"]["x"], item["value"]["y"]
            x= item["value"]["x"]*item["original_width"]/100
            y= item["value"]["y"]*item["original_height"]/100
            label = item["value"]["keypointlabels"][0]

            if label == "Ji":
                new_data["ridge_number"] += 1
                new_data["ridge_coordinate"].append((x, y))
            elif label == "Other":
                new_data["other_number"] += 1
                new_data["other_coordinate"].append((x, y))
            elif label == "Plus":
                new_data["plus_number"] += 1
                new_data["plus_coordinate"].append((x, y))
            elif label == "Pre-plus":
                new_data["pre_plus_number"] += 1
                new_data["pre_plus_coordinate"].append((x, y))

    return new_data

def parse_json_file(file_dict,save_path):
    
    annotation=[]
    file_list=os.listdir(file_dict)
    for file in file_list:
        with open(os.path.join(file_dict,file), 'r') as f:
            data = json.load(f)
        for json_obj in data:
            annotation.append(parse_json(json_obj))

    with open(save_path, 'w') as f:
        json.dump(annotation, f, indent=2)

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # cleansing
    parser.add_argument('--json_file_dict', type=str, default="./json_src",
                        help='Path to the source folder containing original datasets.')
    parser.add_argument('--tar_json_path', type=str, default="./annotation.json",
                        help='Path to the source folder containing original datasets.')
    args = parser.parse_args()

    parse_json_file(args.json_file_dict,save_path=args.tar_json_path)
