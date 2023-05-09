import torch
from PIL import Image, ImageEnhance
from torchvision.transforms import functional as F
from torchvision import transforms

class transformsCompose:
    def __init__(self,transforms):
        self.transforms=transforms
    def __call__(self,x,y):
        for transform in self.transforms:
            x,y=transform(x,y)
        return x,y
        
class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, img, keypoints):
        original_size = img.size
        img = F.resize(img, self.size)
        scale_x = self.size[0] / original_size[0]
        scale_y = self.size[1] / original_size[1]
        keypoints = torch.tensor([[k[0] * scale_x, k[1] * scale_y] for k in keypoints])
        return img, keypoints

class ContrastEnhancement:
    def __init__(self, factor=1.5):
        self.factor = factor

    def __call__(self, img, label):
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(self.factor)
        return img, label
     
class Fix_RandomRotation:
    def __init__(self, degrees=360, resample=False, expand=False, center=None):
        self.degrees = degrees
        self.resample = resample
        self.expand = expand
        self.center = center

    def get_params(self):
        p = torch.rand(1)

        if p >= 0 and p < 0.25:
            angle = -180
        elif p >= 0.25 and p < 0.5:
            angle = -90
        elif p >= 0.5 and p < 0.75:
            angle = 90
        else:
            angle = 0
        return angle

    def __call__(self, img, keypoints):
        angle = self.get_params()
        img = F.rotate(img, angle, F.InterpolationMode.NEAREST, expand=self.expand, center=self.center)
        img_w, img_h = img.size
        rotated_keypoints = []

        for keypoint in keypoints:
            x, y = keypoint
            if angle == -90:
                rotated_keypoints.append(torch.tensor([y, x]))
            elif angle == 90:
                rotated_keypoints.append(torch.tensor([y, img_h - x]))
            elif angle == 180:
                rotated_keypoints.append(torch.tensor([img_w - x, img_h - y]))
            else:
                rotated_keypoints.append(torch.tensor([x, y]))
        
        return img, torch.stack(rotated_keypoints)
    
class RandomHorizontalFlip:
    def __call__(self, img, keypoints):
        if torch.rand(1) < 0.5:
            img = F.hflip(img)
            keypoints = torch.tensor([[img.size[0] - k[0], k[1]] for k in keypoints])
        return img, keypoints

class RandomVerticalFlip:
    def __call__(self, img, keypoints):
        if torch.rand(1) < 0.5:
            img = F.vflip(img)
            keypoints = torch.tensor([[k[0], img.size[1] - k[1]] for k in keypoints])
        return img, keypoints
    
class ToTensor:
    def __init__(self) -> None:
        self.totensor=transforms.ToTensor()
    def __call__(self, img, keypoint):
        img=self.totensor(img)
        return img, keypoint
    

class Normalize:
    def __init__(self,mean,std) -> None:
        self.norm=transforms.Normalize(mean,std)
    def __call__(self, img, keypoint):
        img=self.norm(img)
        return img,keypoint