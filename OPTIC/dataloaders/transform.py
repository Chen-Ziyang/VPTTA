import numpy as np
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.spatial_transforms import SpatialTransform_2, MirrorTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform, ContrastAugmentationTransform, FancyColorTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform


def get_train_transform(patch_size=(512, 512)):
    tr_transforms = []
    tr_transforms.append(
        SpatialTransform_2(
            patch_size, [i // 2 for i in patch_size],
            do_elastic_deform=True, deformation_scale=(0, 0.25),
            do_rotation=True,
            angle_x=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            angle_y=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            do_scale=True, scale=(0.75, 1.25),
            border_mode_data='constant', border_cval_data=0,
            border_mode_seg='constant', border_cval_seg=0,
            order_seg=1, order_data=3,
            random_crop=True,
            p_el_per_sample=0.1, p_rot_per_sample=0.1, p_scale_per_sample=0.1
        )
    )
    tr_transforms.append(MirrorTransform(axes=(0, 1, 2)))
    tr_transforms.append(BrightnessMultiplicativeTransform((0.7, 1.5), per_channel=True, p_per_sample=0.15))
    # tr_transforms.append(ContrastAugmentationTransform((0.75, 1.25), per_channel=True, p_per_sample=0.15))
    # tr_transforms.append(GammaTransform(gamma_range=(0.5, 2), invert_image=True, per_channel=True, p_per_sample=0.15))
    tr_transforms.append(GaussianNoiseTransform(noise_variance=(0, 0.05), p_per_sample=0.15))
    # tr_transforms.append(GaussianBlurTransform(blur_sigma=(0.5, 1.5), different_sigma_per_channel=True,
    #                                            p_per_channel=0.5, p_per_sample=0.15))
    tr_transforms = Compose(tr_transforms)
    return tr_transforms


def collate_fn_w_transform(batch):
    image, label, name = zip(*batch)
    image = np.stack(image, 0)
    label = np.stack(label, 0)
    name = np.stack(name, 0)
    data_dict = {'data': image, 'mask': label, 'name': name}
    tr_transforms = get_train_transform()
    data_dict = tr_transforms(**data_dict)
    data_dict['mask'] = to_one_hot_list(data_dict['mask'])
    return data_dict


def collate_fn_wo_transform(batch):
    image, label, name = zip(*batch)
    image = np.stack(image, 0)
    label = np.stack(label, 0)
    name = np.stack(name, 0)
    data_dict = {'data': image, 'mask': label, 'name': name}
    data_dict['mask'] = to_one_hot_list(data_dict['mask'])
    return data_dict


def to_one_hot_list(mask_list):
    list = []
    for i in range(mask_list.shape[0]):
        mask = to_one_hot(mask_list[i].squeeze(0))
        list.append(mask)
    return np.stack(list, 0)


def to_one_hot(pre_mask, classes=2):
    mask = np.zeros((pre_mask.shape[0], pre_mask.shape[1], classes))
    mask[pre_mask == 1] = [1, 0]
    mask[pre_mask == 2] = [1, 1]
    return mask.transpose(2, 0, 1)

