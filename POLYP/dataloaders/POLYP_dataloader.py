from torch.utils import data
from PIL import Image
from batchgenerators.utilities.file_and_folder_operations import *
import torchvision.transforms as transforms


class POLYP_dataset(data.Dataset):
    def __init__(self, root, img_list, label_list, target_size=512):
        super().__init__()
        self.root = root
        self.img_list = img_list
        self.label_list = label_list
        self.len = len(img_list)
        self.target_size = (target_size, target_size)

        self.img_transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor()])

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        img_file = os.path.join(self.root, self.img_list[item])
        label_file = os.path.join(self.root, self.label_list[item])
        img = Image.open(img_file).convert('RGB')
        label = Image.open(label_file).convert('L')

        img = self.img_transform(img)
        label = self.gt_transform(label)

        return img, label, img_file

