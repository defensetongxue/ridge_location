import json
import os
import random

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

def parse_json_file(file_dict):
    
    annotation=[]
    file_list=sorted(os.listdir(file_dict))
    print(f"read the origianl json file from {file_list}")
    for file in file_list:
        with open(os.path.join(file_dict,file), 'r') as f:
            data = json.load(f)
        for json_obj in data:
            annotation.append(parse_json(json_obj))

    return annotation

def split_data(data_path, annotations, train_proportion=0.7, val_proportion=0.15):
    # random.shuffle(annotations)

    train_split_size = int(train_proportion * len(annotations))
    val_split_size = int(val_proportion * len(annotations))
    train_annotations = annotations[:train_split_size]
    val_annotations = annotations[train_split_size:train_split_size + val_split_size]
    test_annotations = annotations[train_split_size + val_split_size:]

    with open(os.path.join(data_path, 'annotations', 'train.json'), 'w') as f:
        json.dump(train_annotations, f, indent=2)

    with open(os.path.join(data_path, 'annotations', 'val.json'), 'w') as f:
        json.dump(val_annotations, f, indent=2)

    with open(os.path.join(data_path, 'annotations', 'test.json'), 'w') as f:
        json.dump(test_annotations, f, indent=2)

    print(f"Total samples: {len(annotations)}")
    print(f"Train samples: {len(train_annotations)}")
    print(f"Validation samples: {len(val_annotations)}")
    print(f"Test samples: {len(test_annotations)}")

if __name__=='__main__':
    from config import get_config
    args=get_config()
    
    # cleansing
    annotations=parse_json_file(args.json_file_dict)
    split_data(args.path_tar,annotations)
