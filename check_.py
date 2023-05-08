import os
import json
import cv2

def visualize_coordinates(data, data_path):
    # Read the image
    image_path = os.path.join(data_path, "images", data['image_name'])
    print(image_path)
    image = cv2.imread(image_path)

    # Convert the image from BGR to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Draw the coordinates on the image
    for coord in data['ridge_coordinate']:
        x, y = int(coord[0]), int(coord[1])
        cv2.circle(image, (x, y), 5, (255, 0, 0), -1)

    # Save the image with the drawn coordinates
    save_path = os.path.join('./experiments', 'check', data['image_name'])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path,image)

if __name__ =="__main__":
    with open('annotation.json','r') as f:
        data=json.load(f)
    test_cnt=5
    for i in data:
        if test_cnt<=0:
            break
        test_cnt-=1

        visualize_coordinates(i,'../autodl-tmp/dataset_ROP')