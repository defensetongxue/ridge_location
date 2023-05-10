import numpy as np
from segment_anything import sam_model_registry, SamPredictor
class SAMProcesser():
    def __init__(self,threshold=0.02):
        sam_checkpoint = "./SamModule/checkpoint/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        device = "cuda" # have to use cuda to predict
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)

        self.predictor = SamPredictor(sam)
    def show_mask(self, mask, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        return mask_image
    def __call__(self, img,coordinates,save_path=None):
        self.predictor.set_image(img)
        input_points,input_labels=coordinates
        # open the image and preprocess
        masks, scores, logits = self.predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True,
        )       

        mask_image=self.show_mask(masks[0])
        # file the mask through

        return mask_image
