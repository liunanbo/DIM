import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import numpy as np
import torch




class Dataset_Builder_pathlist(Dataset):


    def __init__(self, imgs_path_list, transform=None):
        self.imgs_path_list = imgs_path_list
        self.transform = transform

    def __len__(self):
        return len(self.imgs_path_list)

    def __getitem__(self, idx):


        img_path = self.imgs_path_list[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image

class RandomTranslateWithReflect:
    '''
    Translate image randomly
    Translate vertically and horizontally by n pixels where
    n is integer drawn uniformly independently for each axis
    from [-max_translation, max_translation].
    Fill the uncovered blank area with reflect padding.
    '''

    def __init__(self, max_translation):
        self.max_translation = max_translation

    def __call__(self, old_image):
        xtranslation, ytranslation = np.random.randint(-self.max_translation,
                                                       self.max_translation + 1,
                                                       size=2)
        xpad, ypad = abs(xtranslation), abs(ytranslation)
        xsize, ysize = old_image.size

        flipped_lr = old_image.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_tb = old_image.transpose(Image.FLIP_TOP_BOTTOM)
        flipped_both = old_image.transpose(Image.ROTATE_180)

        new_image = Image.new("RGB", (xsize + 2 * xpad, ysize + 2 * ypad))

        new_image.paste(old_image, (xpad, ypad))

        new_image.paste(flipped_lr, (xpad + xsize - 1, ypad))
        new_image.paste(flipped_lr, (xpad - xsize + 1, ypad))

        new_image.paste(flipped_tb, (xpad, ypad + ysize - 1))
        new_image.paste(flipped_tb, (xpad, ypad - ysize + 1))

        new_image.paste(flipped_both, (xpad - xsize + 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad - xsize + 1, ypad + ysize - 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad + ysize - 1))

        new_image = new_image.crop((xpad - xtranslation,
                                    ypad - ytranslation,
                                    xpad + xsize - xtranslation,
                                    ypad + ysize - ytranslation))
        return new_image

class Image_Transforms:

    def __init__(self,stage='train',input_shape=256):
        # image augmentation functions
        self.stage = stage

        self.flip_lr = transforms.RandomHorizontalFlip(p=0.5)
        rand_crop = transforms.RandomResizedCrop(input_shape, scale=(0.3, 1.0), ratio=(0.7, 1.4),interpolation=3)
        col_jitter = transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.25)
        post_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize((input_shape,input_shape)),
            post_transform
        ])
        self.train_transform = transforms.Compose([
            rand_crop,
            col_jitter,
            rnd_gray,
            post_transform
        ])

    def __call__(self, inp):
        inp = self.flip_lr(inp)
        if self.stage == 'train':
            out = self.train_transform(inp)
        if self.stage == 'test':
            out = self.test_transform(inp)
        return out



def Image_Loader(img_path=None,
                batch_size=256,
                shuffle=False,
                drop_last=False,num_workers=2,input_shape=256,stage = 'train'):

    imgs_path_list= []
    for root, _, files in os.walk(img_path):
        for name in files:
            if name.endswith(('jpg','png')):
                img_path = os.path.join(root, name)
                imgs_path_list.append(img_path)

    


    # Preprocessing transformation
    transform= Image_Transforms(stage=stage,input_shape=input_shape)
    # Build torch dataset
    dataset = Dataset_Builder_pathlist(transform=transform,imgs_path_list= imgs_path_list)
    loader = torch.utils.data.DataLoader(dataset, 
                                            batch_size=batch_size, 
                                            shuffle=shuffle, 
                                            drop_last=drop_last, 
                                            num_workers=num_workers)
    loader.img_path = imgs_path_list

    return loader