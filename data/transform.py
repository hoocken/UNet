import random
import numpy as np
from torchvision import transforms
from torchvision.transforms.v2 import functional
import abc

    
class ComposeTransform:
    """Transform class that combines multiple other transforms into one"""
    def __init__(self, transforms):
        """
        :param transforms: transforms to be combined
        """
        self.transforms = transforms

    def __call__(self, images, seg):
        for transform in self.transforms:
            images, seg = transform(images, seg)
        return images, seg


class RandomHorizontalFlip:
    """
    Transform class that flips an image horizontically randomly with a given probability.
    """

    def __init__(self, prob=0.5):
        """
        Parameters:
            prob: Probability of the image being flipped
        """
        self.p = prob

    def __call__(self, image, seg):
        """
        Flip the image and segmentation correspondingly

        Parameters:
            image: ndarray of shape (H, W)
            seg: ndarray of shape (C, H, W)
        
        Returns:
            Transformed image and segmentation
        """
        rand = random.uniform(0,1)
        if rand < self.p:
            image = functional.hflip(image)
            seg = functional.hflip(seg)

        return image, seg
    
class RandomInvert:
    """
    Transform class that flips an image horizontically randomly with a given probability.
    """

    def __init__(self, prob=0.5):
        """
        Parameters:
            prob: Probability of the image being flipped
        """
        self.p = prob

    def __call__(self, image, seg):
        """
        Flip the image and segmentation correspondingly

        Parameters:
            image: ndarray of shape (H, W)
            seg: ndarray of shape (C, H, W)
        
        Returns:
            Transformed image and segmentation
        """
        rand = random.uniform(0,1)
        if rand < self.p:
            image = functional.invert(image)

        return image, seg

class RandomRotation:
    def __init__(self, rotation=30):
        self.rotation = rotation
        pass

    def __call__(self, image, seg):
        params = transforms.RandomRotation.get_params([-self.rotation, self.rotation])
        image = functional.rotate(image, params)
        seg = functional.rotate(seg, params)
        return image, seg
    
class GaussianBlur:
    def __init__(self, kernel_size=5, range=[0.1, 2]):
        self.kernel_size = kernel_size
        self.range = range

    def __call__(self, image, seg):
        image = functional.gaussian_blur(image, self.kernel_size, self.range)

        return image, seg
    
class GaussianNoise:
    def __init__(self, noise=0.05):
        self.noise = noise

    def __call__(self, image, seg):
        image = functional.gaussian_noise(image, 0, self.noise)

        return image, seg

class RandomCrop:
    def __init__(self, min_crop_size=0.3, translation_range=[0.75, 1.25]):
        """
        Parameters:
            min_crop_size: Minimal size proportion that an image can be cropped to, before being resized
        """
        self.min_crop_size = min_crop_size
        self.translation_range = translation_range

    def __call__(self, image, seg):
        """
        Crop the image and segmentation correspondingly, then resized to
        the original shape

        Parameters:
            image: tensor of shape (N, C, H, W)
            seg: tensor of shape (N, C, H, W)
        
        Returns:
            Transformed image and segmentation
        """
        # rand = random.uniform(0,1)
        # if rand < self.p:
        _, _, H, W = image.shape

        params = transforms.RandomResizedCrop.get_params(image, [self.min_crop_size, 1], self.translation_range)
            
            # size = random.randint(int(self.min_crop_size * H), H)
            # left = random.randint(0, W - size)
            # up = random.randint(0, H - size)

            # cropped_image = image[:, left:left+size, up:up+size]
            # cropped_seg = seg[:, :, left:left+size, up:up+size]

        image = functional.resized_crop(image, *params, size=[H, W])
        seg = functional.resized_crop(seg, *params, size=[H, W])

        return image, seg
    
class RandomResize:
    def __init__(self, min_resize=0.6):
        """
        Parameters:
            min_resize: Minimal size proportion that an image can be cropped to, before being resized
        """
        self.min_resize = min_resize

    def __call__(self, image, seg):
        """
        Resize the image, then pad it with blanks

        Parameters:
            image: tensor of shape (N, C, H, W)
            seg: tensor of shape (N, C, H, W)
        
        Returns:
            Transformed image and segmentation
        """
        _, _, H, W = image.shape
        rand_size = random.randint(int(self.min_resize * H), H)

        image = functional.resize(image, size=[rand_size, rand_size])
        seg = functional.resize(seg, size=[rand_size, rand_size])

        image = functional.pad(image, H - rand_size)
        seg = functional.pad(seg, H - rand_size)

        return image, seg
    
class RandomAffine:
    def __init__(self, degrees=30, min_resize=0.8, translation=0.1):
        """
        Applies an affine transform
        Parameters:
            degrees: Degree range for rotation
            min_resize: Minimal size to be resized
            translation: Translation
        """
        self.degrees = degrees
        self.min_resize = min_resize
        self.translation = translation

    def __call__(self, image, seg):
        """
        Applies an affine transformation through rotation, translation, and resize

        Parameters:
            image: tensor of shape (N, C, H, W)
            seg: tensor of shape (N, C, H, W)
        
        Returns:
            Transformed image and segmentation
        """
        _, _, H, W = image.shape

        params = transforms.RandomAffine.get_params(img_size=[H, W], shears=None, degrees=[-self.degrees, self.degrees], translate=[self.translation, self.translation], scale_ranges=[self.min_resize, 1])

        image = functional.affine(image, *params)
        seg = functional.affine(seg, *params)

        return image, seg