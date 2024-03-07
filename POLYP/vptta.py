import os
import torch
import numpy as np
import argparse, sys, datetime
from config import Logger
from torch.autograd import Variable
from utils.metrics import calculate_metrics
from utils.convert import AdaBN
from utils.memory import Memory
from utils.prompt import Prompt
from networks.PraNet_Res2Net_TTA import PraNet
from torch.utils.data import DataLoader
from dataloaders.POLYP_dataloader import POLYP_dataset
from dataloaders.convert_csv_to_list import convert_labeled_list


torch.set_num_threads(1)


class VPTTA:
    def __init__(self, config):
        # Save Log
        time_now = datetime.datetime.now().__format__("%Y%m%d_%H%M%S_%f")
        log_root = os.path.join(config.path_save_log, 'VPTTA')
        if not os.path.exists(log_root):
            os.makedirs(log_root)
        log_path = os.path.join(log_root, time_now + '.log')
        sys.stdout = Logger(log_path, sys.stdout)

        # Data Loading
        target_test_csv = []
        for target in config.Target_Dataset:
            target_test_csv.append(target + '_train.csv')
            target_test_csv.append(target + '_test.csv')
        ts_img_list, ts_label_list = convert_labeled_list(config.dataset_root, target_test_csv)
        target_test_dataset = POLYP_dataset(config.dataset_root, ts_img_list, ts_label_list,
                                            config.image_size)
        self.target_test_loader = DataLoader(dataset=target_test_dataset,
                                             batch_size=config.batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             drop_last=False,
                                             num_workers=config.num_workers)
        self.image_size = config.image_size

        # Model
        self.load_model = os.path.join(config.model_root, str(config.Source_Dataset))  # Pre-trained Source Model
        self.in_ch = config.in_ch
        self.out_ch = config.out_ch

        # Optimizer
        self.optim = config.optimizer
        self.lr = config.lr
        self.momentum = config.momentum
        self.betas = (config.beta1, config.beta2)
        self.weight_decay = config.weight_decay

        # GPU
        self.device = config.device

        # Warm-up
        self.warm_n = config.warm_n

        # Prompt
        self.prompt_alpha = config.prompt_alpha
        self.iters = config.iters

        # Initialize the pre-trained model and optimizer
        self.build_model()

        # Memory Bank
        self.neighbor = config.neighbor
        self.memory_bank = Memory(size=config.memory_size, dimension=self.prompt.data_prompt.numel())

        # Print Information
        for arg, value in vars(config).items():
            print(f"{arg}: {value}")
        self.print_prompt()
        print('***' * 20)

    def build_model(self):
        self.prompt = Prompt(prompt_alpha=self.prompt_alpha, image_size=self.image_size).to(self.device)
        self.model = PraNet(warm_n=self.warm_n).to(self.device)
        checkpoint = torch.load(os.path.join(self.load_model, 'pretrain-PraNet.pth'))
        self.model.load_state_dict(checkpoint, strict=True)

        if self.optim == 'SGD':
            self.optimizer = torch.optim.SGD(
                self.prompt.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                nesterov=True,
                weight_decay=self.weight_decay
            )
        elif self.optim == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.prompt.parameters(),
                lr=self.lr,
                betas=self.betas,
                weight_decay=self.weight_decay
            )

    def print_prompt(self):
        num_params = 0
        for p in self.prompt.parameters():
            num_params += p.numel()
        print("The number of total parameters: {}".format(num_params))

    def run(self):
        metrics_test = [[], [], []]
        metric_dict = ['Dice', 'Enhanced_Align', 'Structure_Measure']

        # Valid on Target
        for batch, data in enumerate(self.target_test_loader):
            x, y, path = data
            x, y = Variable(x).to(self.device), Variable(y).to(self.device)

            self.model.eval()
            self.prompt.train()
            self.model.change_BN_status(new_sample=True)

            # Initialize Prompt
            if len(self.memory_bank.memory.keys()) >= self.neighbor:
                _, low_freq = self.prompt(x)
                init_data = self.memory_bank.get_neighbours(keys=low_freq.cpu().numpy(), k=self.neighbor)
            else:
                init_data = torch.ones((1, 3, self.prompt.prompt_size, self.prompt.prompt_size)).data
            self.prompt.update(init_data)

            # Train Prompt for n iters (1 iter in our VPTTA)
            for tr_ites in range(self.iters):
                prompt_x, _ = self.prompt(x)
                _ = self.model(prompt_x)

                times, bn_loss = 0, 0
                for nm, m in self.model.resnet.named_modules():
                    if isinstance(m, AdaBN):
                        bn_loss += m.bn_loss
                        times += 1
                loss = bn_loss / times

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.model.change_BN_status(new_sample=False)

            # Inference
            self.model.eval()
            self.prompt.eval()
            with torch.no_grad():
                prompt_x, low_freq = self.prompt(x)
                pred_logit = self.model(prompt_x)

            # Update the Memory Bank
            self.memory_bank.push(keys=low_freq.cpu().numpy(), logits=self.prompt.data_prompt.detach().cpu().numpy())

            # Calculate the metrics
            seg_output = torch.sigmoid(pred_logit)
            metrics = calculate_metrics(seg_output.detach().cpu(), y.detach().cpu())
            for i in range(len(metrics)):
                assert isinstance(metrics[i], list), "The metrics value is not list type."
                metrics_test[i] += metrics[i]

        test_metrics_y = np.mean(metrics_test, axis=1)
        print_test_metric_mean = {}
        for i in range(len(test_metrics_y)):
            print_test_metric_mean[metric_dict[i]] = test_metrics_y[i]
        print("Test Metrics Mean: ", print_test_metric_mean)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument('--Source_Dataset', type=str, default='BKAI',
                        help='BKAI/CVC-ClinicDB/ETIS-LaribPolypDB/Kvasir-SEG')
    parser.add_argument('--Target_Dataset', type=list)

    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--image_size', type=int, default=352)

    # Model
    parser.add_argument('--in_ch', type=int, default=3)
    parser.add_argument('--out_ch', type=int, default=1)

    # Optimizer
    parser.add_argument('--optimizer', type=str, default='Adam', help='SGD/Adam/AdamW')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.99)  # momentum in SGD
    parser.add_argument('--beta1', type=float, default=0.9)  # beta1 in Adam
    parser.add_argument('--beta2', type=float, default=0.99)  # beta2 in Adam
    parser.add_argument('--weight_decay', type=float, default=0.00)

    # Training
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--iters', type=int, default=1)

    # Hyperparameters in memory bank, prompt, and warm-up statistics
    parser.add_argument('--memory_size', type=int, default=40)
    parser.add_argument('--neighbor', type=int, default=16)
    parser.add_argument('--prompt_alpha', type=float, default=0.01)
    parser.add_argument('--warm_n', type=int, default=5)

    # Path
    parser.add_argument('--path_save_log', type=str, default='./logs/')
    parser.add_argument('--model_root', type=str, default='./models/')
    parser.add_argument('--dataset_root', type=str, default='/media/userdisk0/zychen/Datasets/Polyp')

    # Cuda (default: the first available device)
    parser.add_argument('--device', type=str, default='cuda:0')

    config = parser.parse_args()

    config.Target_Dataset = ['BKAI', 'CVC-ClinicDB', 'ETIS-LaribPolypDB', 'Kvasir-SEG']
    config.Target_Dataset.remove(config.Source_Dataset)

    TTA = VPTTA(config)
    TTA.run()
