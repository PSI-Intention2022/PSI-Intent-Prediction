import torch
import os
import numpy as np
from torchvision import transforms
import cv2
import PIL
from PIL import Image
import copy
from torch.utils.data.sampler import WeightedRandomSampler



class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, data, args, stage='train'):
        super(VideoDataset, self).__init__()
        # ['frame', 'bbox', 'binary_intent', 'reason_feats', 'ped_id', 'video_id']
        self.data = data
        self.args = args
        self.stage = stage
        self.set_transform()
        self.images_path = os.path.join(args.dataset_root_path, 'frames')
        print(self.data.keys())

    def __getitem__(self, index):
        video_ids = self.data['video_id'][index]
        ped_ids = self.data['ped_id'][index]
        assert video_ids[0] == video_ids[-1] # all video_id should be the same
        assert ped_ids[0] == ped_ids[-1]  # all video_id should be the same
        frame_list = self.data['frame'][index][:self.args.observe_length] # return first 15 frames as observed

        bboxes = self.data['bbox'][index] # return all 60 frames #[:-1] # take first 15 frames as input

        intention_binary = self.data['intention_binary'][index] # all frames intentions are returned
        intention_prob = self.data['intention_prob'][index] # all frames, 3-dimension votes probability

        # reason = [np.array(ele) for ele in self.data['reason_feats'][index]]
        # reason_origin = np.array(reason) # reason of count # all 60 frames reasons are returned
        # reason = np.array([[1 if r > 0 else 0 for r in rs] for rs in reason]) #reason > 0 # change reason to binary{0, 1}
        #
        disagree_score = self.data['disagree_score'][index] # all scores for all frames are returned

        assert len(bboxes) == self.args.max_track_size # following bboxes are used to calculate trajectory
        assert len(frame_list) == self.args.observe_length # only 15 frames is necessary

        # if self.args.load_image:
        #     images, cropped_images = self.load_images(video_id, frame_list, bboxes)
        # else:
        #     images, cropped_images = [], []

        global_featmaps, local_featmaps = self.load_features(video_ids, ped_ids, frame_list)
        reason_features = self.load_reason_features(video_ids, ped_ids, frame_list)

        for f in range(len(frame_list)): #(len(bboxes)):
            box = bboxes[f]
            xtl, ytl, xrb, yrb = box

            if self.args.task_name == 'ped_intent' or self.args.task_name == 'ped_traj':
                bboxes[f] = [xtl, ytl, xrb, yrb]


        if self.args.normalize_bbox == 'L2':
            raise Exception("Bboxes nromalize is not defined!")
        elif self.args.normalize_bbox == 'subtract_first_frame':
            bboxes = bboxes - bboxes[:1, :] # minus the first frame bbox positions
        else:
            pass



        data = {
            # 'cropped_images': cropped_images,
            # 'images': images,
            'local_featmaps': local_featmaps,
            'global_featmaps': global_featmaps,
            'bboxes': bboxes,
            # 'intention_onehot': intention_onehot,
            'intention_binary': intention_binary,
            'intention_prob': intention_prob,
            'reason_feats': reason_features,
            # 'reason': reason,
            # 'reason_origin': reason_origin,
            'frames': np.array([int(f) for f in frame_list]),
            'video_id': video_ids[0], #int(video_id[0][0].split('_')[1])
            'ped_id': ped_ids[0],
            'disagree_score': disagree_score
        }

        return data

    def __len__(self):
        return len(self.data['frame'])

    ''' All below are util functions '''
    def load_reason_features(self, video_ids, ped_ids, frame_list):
        feat_list = []
        video_name = video_ids[0]
        if 'rsn' in self.args.model_name:
            for i in range(len(frame_list)):
                fid = frame_list[i]
                pid = ped_ids[i]
                local_path = os.path.join(self.args.dataset_root_path, 'features/bert_description',
                                          video_name, pid)
                feat_file = np.load(local_path + f'/{fid:03d}.npy')
                feat_list.append(torch.tensor(feat_file))

        feat_list = [] if len(feat_list) < 1 else torch.stack(feat_list)
        return feat_list


    def load_features(self, video_ids, ped_ids, frame_list):
        global_featmaps = []
        local_featmaps = []
        video_name = video_ids[0]
        if 'global' in self.args.model_name:
            for i in range(len(frame_list)):
                fid = frame_list[i]
                pid = ped_ids[i]

                glob_path = os.path.join(self.args.dataset_root_path, 'features', self.args.backbone, 'global_feats', video_name)
                glob_featmap = np.load(glob_path + f'/{fid:03d}.npy')
                global_featmaps.append(torch.tensor(glob_featmap))

        if 'ctxt' in self.args.model_name:
            for i in range(len(frame_list)):
                fid = frame_list[i]
                pid = ped_ids[i]
                local_path = os.path.join(self.args.dataset_root_path, 'features', self.args.backbone, 'context_feats',
                                          video_name, pid)
                local_featmap = np.load(local_path + f'/{fid:03d}.npy')
                local_featmaps.append(torch.tensor(local_featmap))

        global_featmaps = [] if len(global_featmaps) < 1 else torch.stack(global_featmaps)
        local_featmaps = [] if len(local_featmaps) < 1 else torch.stack(local_featmaps)

        return global_featmaps, local_featmaps


    def load_images(self, video_ids, frame_list, bboxes):
        images = []
        cropped_images = []
        video_name = video_ids[0]

        for i in range(len(frame_list)):
            frame_id = frame_list[i]
            bbox = bboxes[i]
            # load original image
            # print(video_id, frame_list, video_name, frame_id, bbox)
            img_path = os.path.join(self.images_path, video_name, str(frame_id).zfill(5)+'.png')
            # print(img_path)
            img = self.rgb_loader(img_path)
            print(img.shape)# 2048 x 2048 x 3 --> 1280 x 720
            # print("Original image size: ", img.shape, bbox)
            # Image.fromarray(img).show()
            # img.shape: H x W x C, RGB channel
            # crop pedestrian surrounding image
            ori_bbox = copy.deepcopy(bbox)

            bbox = self.jitter_bbox(img, [bbox], self.args.crop_mode, 2.0)[0]

            # x1, y1, x2, y2 = bbox

            bbox = self.squarify(bbox, 1, img.shape[1])
            bbox = list(map(int, bbox[0:4]))

            cropped_img = Image.fromarray(img).crop(bbox)
            cropped_img = np.array(cropped_img)
            if not cropped_img.shape:
                print("Error in crop: ", video_id[0][0], frame_id, ori_bbox, bbox)
            cropped_img = self.img_pad(cropped_img, mode='pad_resize', size=224) # return PIL.image type

            cropped_img = np.array(cropped_img)
            # cv2.imshow(str(i), np.array(cropped_img))
            # cv2.waitKey(1000)
            # cv2.destroyAllWindows()

            if self.transform:
                # print("before transform - img: ", img.shape, " cropped: ", cropped_img.shape)
                img = self.transform(img)
                cropped_img = self.transform(cropped_img)
                # print("after transform - img: ", img.shape, " cropped: ", cropped_img.shape)
                # After transform, changed to tensor, img.shape: C x H x W
            images.append(img)
            cropped_images.append(cropped_img)

        return torch.stack(images), torch.stack(cropped_images) # Time x Channel x H x W


    def rgb_loader(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def set_transform(self):
        if self.stage == 'train':
            resize_size = 256
            crop_size = 224
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((resize_size, resize_size)),
                transforms.RandomCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            resize_size = 256
            crop_size = 224
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((resize_size, resize_size)),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def squarify(self, bbox, squarify_ratio, img_width):
        width = abs(bbox[0] - bbox[2])
        height = abs(bbox[1] - bbox[3])
        width_change = height * squarify_ratio - width
        # width_change = float(bbox[4])*self._squarify_ratio - float(bbox[3])
        bbox[0] = bbox[0] - width_change/2
        bbox[2] = bbox[2] + width_change/2
        # bbox[1] = str(float(bbox[1]) - width_change/2)
        # bbox[3] = str(float(bbox[3]) + width_change)
        # Squarify is applied to bounding boxes in Matlab coordinate starting from 1
        if bbox[0] < 0:
            bbox[0] = 0

        # check whether the new bounding box goes beyond image boarders
        # If this is the case, the bounding box is shifted back
        if bbox[2] > img_width:
            # bbox[1] = str(-float(bbox[3]) + img_dimensions[0])
            bbox[0] = bbox[0]-bbox[2] + img_width
            bbox[2] = img_width
        return bbox

    def jitter_bbox(self, img, bbox, mode, ratio):
        '''
        This method jitters the position or dimentions of the bounding box.
        mode: 'same' returns the bounding box unchanged
              'enlarge' increases the size of bounding box based on the given ratio.
              'random_enlarge' increases the size of bounding box by randomly sampling a value in [0,ratio)
              'move' moves the center of the bounding box in each direction based on the given ratio
              'random_move' moves the center of the bounding box in each direction by randomly sampling a value in [-ratio,ratio)
        ratio: The ratio of change relative to the size of the bounding box. For modes 'enlarge' and 'random_enlarge'
               the absolute value is considered.
        Note: Tha ratio of change in pixels is calculated according to the smaller dimension of the bounding box.
        '''
        assert (mode in ['same', 'enlarge', 'move', 'random_enlarge', 'random_move']), \
            'mode %s is invalid.' % mode

        if mode == 'same':
            return bbox

        # img = self.rgb_loader(img_path)
        # img_width, img_heigth = img.size

        if mode in ['random_enlarge', 'enlarge']:
            jitter_ratio = abs(ratio)
        else:
            jitter_ratio = ratio

        if mode == 'random_enlarge':
            jitter_ratio = np.random.random_sample() * jitter_ratio
        elif mode == 'random_move':
            # for ratio between (-jitter_ratio, jitter_ratio)
            # for sampling the formula is [a,b), b > a,
            # random_sample * (b-a) + a
            jitter_ratio = np.random.random_sample() * jitter_ratio * 2 - jitter_ratio

        jit_boxes = []
        for b in bbox:
            bbox_width = b[2] - b[0]
            bbox_height = b[3] - b[1]

            width_change = bbox_width * jitter_ratio
            height_change = bbox_height * jitter_ratio

            if width_change < height_change:
                height_change = width_change
            else:
                width_change = height_change

            if mode in ['enlarge', 'random_enlarge']:
                b[0] = b[0] - width_change // 2
                b[1] = b[1] - height_change // 2
            else:
                b[0] = b[0] + width_change // 2
                b[1] = b[1] + height_change // 2

            b[2] = b[2] + width_change // 2
            b[3] = b[3] + height_change // 2

            # Checks to make sure the bbox is not exiting the image boundaries
            b = self.bbox_sanity_check(img, b)
            jit_boxes.append(b)
        # elif crop_opts['mode'] == 'border_only':
        return jit_boxes

    def bbox_sanity_check(self, img, bbox):
        '''
        This is to confirm that the bounding boxes are within image boundaries.
        If this is not the case, modifications is applied.
        This is to deal with inconsistencies in the annotation tools
        '''

        img_heigth, img_width, channel = img.shape
        if bbox[0] < 0:
            bbox[0] = 0.0
        if bbox[1] < 0:
            bbox[1] = 0.0
        if bbox[2] >= img_width:
            bbox[2] = img_width - 1
        if bbox[3] >= img_heigth:
            bbox[3] = img_heigth - 1
        return bbox

    def img_pad(self, img, mode='warp', size=224):
        '''
        Pads a given image.
        Crops and/or pads a image given the boundries of the box needed
        img: the image to be coropped and/or padded
        bbox: the bounding box dimensions for cropping
        size: the desired size of output
        mode: the type of padding or resizing. The modes are,
            warp: crops the bounding box and resize to the output size
            same: only crops the image
            pad_same: maintains the original size of the cropped box  and pads with zeros
            pad_resize: crops the image and resize the cropped box in a way that the longer edge is equal to
            the desired output size in that direction while maintaining the aspect ratio. The rest of the image is
            padded with zeros
            pad_fit: maintains the original size of the cropped box unless the image is biger than the size in which case
            it scales the image down, and then pads it
        '''
        assert (mode in ['same', 'warp', 'pad_same', 'pad_resize', 'pad_fit']), 'Pad mode %s is invalid' % mode
        image = img.copy()
        if mode == 'warp':
            warped_image = image.resize((size, size), PIL.Image.NEAREST)
            return warped_image
        elif mode == 'same':
            return image
        elif mode in ['pad_same', 'pad_resize', 'pad_fit']:
            img_size = (image.shape[0], image.shape[1]) # size is in (width, height)
            ratio = float(size) / max(img_size)
            if mode == 'pad_resize' or \
                    (mode == 'pad_fit' and (img_size[0] > size or img_size[1] > size)):
                img_size = (int(img_size[0] * ratio), int(img_size[1] * ratio))# tuple([int(img_size[0] * ratio), int(img_size[1] * ratio)])
                # print(img_size, type(img_size), type(img_size[0]), type(img_size[1]))
                # print(type(image), image.shape)
                try:
                    image = Image.fromarray(image)
                    image = image.resize(img_size, PIL.Image.NEAREST)
                except Exception as e:
                    print("Error from np-array to Image: ", image.shape)
                    print(e)

            padded_image = PIL.Image.new("RGB", (size, size))
            padded_image.paste(image, ((size - img_size[0]) // 2,
                                       (size - img_size[1]) // 2))
            return padded_image
