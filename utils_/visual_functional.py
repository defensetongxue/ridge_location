import torch
import math
import cv2

def visualize_and_save_landmarks(image_path, preds, save_path,text=None):
    img = cv2.imread(image_path)
    img=cv2.resize(img,(416,416))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Ensure preds is a NumPy array with shape (98, 2)
    if isinstance(preds, torch.Tensor):
        preds = preds.squeeze(0).numpy()

    # Draw landmarks on the image
    print(text)
    for x, y in preds:
        cv2.circle(img, (int(x), int(y)), 8, (255, 0, 0), -1)
        if text:
            cv2.putText(img,str(text),(200, 200),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(255, 0, 0),thickness=4)
    # Save the image with landmarks
    cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
def get_preds(scores):
    """
    get predictions from score maps in torch Tensor
    return type: torch.LongTensor
    """
    assert scores.dim() == 4, 'Score maps should be 4-dim'
    maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)

    maxval = maxval.view(scores.size(0), scores.size(1), 1)
    idx = idx.view(scores.size(0), scores.size(1), 1) + 1

    preds = idx.repeat(1, 1, 2).float()

    preds[:, :, 0] = (preds[:, :, 0] - 1) % scores.size(3) + 1
    preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / scores.size(3)) + 1

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds *= pred_mask
    return preds

def decode_preds(output):
    map_width,map_height=output.shape[-2],output.shape[-1]
    coords = get_preds(output)  # float type
    coords = coords.cpu()
    # pose-processing
    for n in range(coords.size(0)):
        for p in range(coords.size(1)):
            hm = output[n][p]
            px = int(math.floor(coords[n][p][0]))
            py = int(math.floor(coords[n][p][1]))
            if (px > 1) and (px < map_width) and (py > 1) and (py < map_height):
                diff = torch.Tensor([hm[py - 1][px] - hm[py - 1][px - 2], hm[py][px - 1]-hm[py - 2][px - 1]])
                coords[n][p] += diff.sign() *.25
    preds = coords.clone()

    # Transform back
    if preds.dim() < 3:
        preds = preds.view(1, preds.size())

    return preds*4 # heatmap is 1/4 to original image


