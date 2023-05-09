import torch
import math
import cv2

def visualize_and_save_landmarks(image_path, preds, maxvals, save_path,text=False):
    print(image_path)
    img = cv2.imread(image_path)
    img = cv2.resize(img, (800,800))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Ensure preds and maxvals are NumPy arrays
    if isinstance(preds, torch.Tensor):
        preds = preds.squeeze(0).numpy()
    if isinstance(maxvals, torch.Tensor):
        maxvals = maxvals.squeeze().numpy()
    # Draw landmarks on the image
    cnt=1
    for pred, maxval in zip(preds, maxvals):
        x, y = pred
        cv2.circle(img, (int(x), int(y)), 8, (255, 0, 0), -1)
        if text:
            cv2.putText(img, f"{maxval:.2f}", (int(x), int(y)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(img, f"{cnt}", (int(x), int(y)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cnt+=1
    # Save the image with landmarks
    cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def get_preds(scores, number, r=20):
    """
    input scores is 2d-tensor heatmap
    number is the number of select point

    return shape should be pres=(number,2) maxvals=(number)
    """

    preds_list = []
    maxvals_list = []

    temp_scores = scores.clone()

    for _ in range(number):
        maxval, idx = torch.max(temp_scores.view(-1), dim=0)
        maxval = maxval.flatten()
        idx = idx.view(-1, 1) + 1

        pred = idx.repeat(1, 2).float()
        pred[:, 0] = (pred[:, 0] - 1) % scores.size(1) + 1
        pred[:, 1] = torch.floor((pred[:, 1] - 1) / scores.size(1)) + 1

        maxvals_list.append(maxval.item())
        preds_list.append(pred.squeeze())
        # Clear the square region around the point
        x, y = int(pred[0, 0].item()), int(pred[0, 1].item())
        xmin, ymin = max(0, x - r // 2), max(0, y - r // 2)
        xmax, ymax = min(scores.size(1), x + r // 2), min(scores.size(0), y + r // 2)
        temp_scores[ymin:ymax, xmin:xmax] = 0

    preds = torch.stack(preds_list)
    maxvals = torch.tensor(maxvals_list)
    return preds, maxvals


def decode_preds(output, visual_num=3):

    assert output.dim() == 4, 'Score maps should be 4-dim'
    assert output.shape[0]==1 ,'visual should be batch==1'
    output=output.squeeze() # (width, height)
    map_width, map_height = output.shape[-2], output.shape[-1]
    coords,maxval = get_preds(output, visual_num)  # float type

    # pose-processing
    for k in range(coords.shape[0]):
        hm = output.clone()
        px = int(math.floor(coords[k][0]))
        py = int(math.floor(coords[k][1]))
        if (px > 1) and (px < map_width) and (py > 1) and (py < map_height):
            diff = torch.Tensor([hm[py - 1][px] - hm[py - 1][px - 2], hm[py][px - 1] - hm[py - 2][px - 1]])
            coords[k] += diff.sign() * 0.25
    preds = coords.clone()

    # Transform back
    return preds * 4,maxval  # heatmap is 1/4 of the original image

