def normalize_image(img_npy):
    """
    :param img_npy: b, c, h, w
    """
    for b in range(img_npy.shape[0]):
        for c in range(img_npy.shape[1]):
            img_npy[b, c] = (img_npy[b, c] - img_npy[b, c].mean()) / img_npy[b, c].std()
    return img_npy


def normalize_image_to_0_1(img):
    if len(img.shape) == 4:
        for b in range(img.shape[0]):
            img[b] = (img[b]-img[b].min())/(img[b].max()-img[b].min())
    else:
        img = (img-img.min())/(img.max()-img.min())
    return img


def normalize_image_to_m1_1(img):
    return -1 + 2 * (img-img.min())/(img.max()-img.min())


# def normalize_image_to_0_1(img):
#     for b in range(img.shape[0]):
#         img[b] = (img[b]-img[b].min())/(img[b].max()-img[b].min())
#     return img
#
#
# def normalize_image_to_m1_1(img):
#     for b in range(img.shape[0]):
#         img[b] = -1 + 2 * (img[b]-img[b].min())/(img[b].max()-img[b].min())
#     return img
