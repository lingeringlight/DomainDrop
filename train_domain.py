# from torch.utils.tensorboard import SummaryWriter
import argparse
# import torch
from torch import nn
from data import data_helper
from models import model_factory
from optimizer.optimizer_helper import get_optim_and_scheduler, get_optim_and_scheduler_scatter
from utils.Logger import Logger
from models.resnet_domain import resnet18, resnet50
import os
import random
import time
from utils.tools import *
from loss.KL_Loss import compute_kl_loss


def get_args():
    parser = argparse.ArgumentParser(description="Script to launch jigsaw training",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--target", default=0, type=int, help="Target")
    parser.add_argument("--device", type=int, default=0, help="GPU num")
    parser.add_argument("--time", default=0, type=int, help="train time")

    parser.add_argument("--eval", default=0, type=int, help="Eval trained models")
    parser.add_argument("--eval_model_path", default="/model/path", help="Path of trained models")

    parser.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size")
    parser.add_argument("--image_size", type=int, default=224, help="Image size")

    parser.add_argument("--data", default="PACS")
    parser.add_argument("--data_root", default="/data/DataSets/")

    parser.add_argument("--KL_Loss", default=1, type=int, help="whether to use consistency of dropout")
    parser.add_argument("--KL_Loss_weight", default=1.5, type=float, help="weight of KL_Loss")
    parser.add_argument("--KL_Loss_T", default=5, type=float, help="T of KL_Loss")

    parser.add_argument("--layer_wise_prob", default=0.8, type=float, help="prob to use layer-wise dropout")

    parser.add_argument("--domain_discriminator_flag", default=1, type=int, help="whether use domain discriminator.")
    parser.add_argument("--domain_loss_flag", default=1, type=int, help="whether use domain loss.")
    parser.add_argument("--discriminator_layers", default=[1, 2, 3, 4], nargs="+", type=int, help="where to place discriminators")
    parser.add_argument("--grl", default=1, type=int, help="whether to use grl")
    parser.add_argument("--lambd", default=0.25, type=float, help="weight of grl")

    parser.add_argument("--drop_percent", default=0.33,  type=float, help="percent of dropped filters")
    parser.add_argument("--filter_WRS_flag", default=1, type=int, help="Weighted Random Selection.")
    parser.add_argument("--recover_flag", default=1, type=int)
    parser.add_argument("--result_path", default="/data/DomainDropout_results/", help="")

    # data aug stuff
    parser.add_argument("--learning_rate", "-l", type=float, default=.002, help="Learning rate")
    parser.add_argument("--epochs", "-e", type=int, default=50, help="Number of epochs")
    parser.add_argument("--min_scale", default=0.8, type=float, help="Minimum scale percent")
    parser.add_argument("--max_scale", default=1.0, type=float, help="Maximum scale percent")
    parser.add_argument("--gray_flag", default=1, type=int, help="whether use random gray")
    parser.add_argument("--random_horiz_flip", default=0.5, type=float, help="Chance of random horizontal flip")
    parser.add_argument("--jitter", default=0.4, type=float, help="Color jitter amount")
    parser.add_argument("--tile_random_grayscale", default=0.1, type=float,
                        help="Chance of randomly greyscaling a tile")
    parser.add_argument("--limit_source", default=None, type=int,
                        help="If set, it will limit the number of training samples")
    parser.add_argument("--limit_target", default=None, type=int,
                        help="If set, it will limit the number of testing samples")
    parser.add_argument("--network", choices=model_factory.nets_map.keys(), help="Which network to use",
                        default="resnet18")
    parser.add_argument("--tf_logger", type=bool, default=True, help="If true will save tensorboard compatible logs")
    parser.add_argument("--folder_name", default='test', help="Used by the logger to save logs")
    parser.add_argument("--bias_whole_image", default=0.9, type=float,
                        help="If set, will bias the training procedure to show more often the whole image")
    parser.add_argument("--TTA", type=bool, default=False, help="Activate test time data augmentation")
    parser.add_argument("--classify_only_sane", default=False, type=bool,
                        help="If true, the network will only try to classify the non scrambled images")
    parser.add_argument("--train_all", default=True, type=bool, help="If true, all network weights will be trained")
    parser.add_argument("--suffix", default="", help="Suffix for the logger")
    parser.add_argument("--nesterov", default=True, type=bool, help="Use nesterov")

    return parser.parse_args()


def get_results_path(args):
    # Make the directory to store the experimental results
    base_result_path = args.result_path + "/" + args.data + "/"
    base_result_path += args.network

    if args.domain_discriminator_flag == 1:
        base_result_path += "_DomainDrop"

        base_result_path += "_layer_wise" + str(args.layer_wise_prob)

        if args.grl == 1:
            base_result_path += "_grl" + str(args.lambd)
        base_result_path += "_channel"

        base_result_path += "_L"
        for i, layer in enumerate(args.discriminator_layers):
            base_result_path += str(layer)
        base_result_path += "_dropP" + str(args.drop_percent)
        base_result_path += "_domain"
        if args.filter_WRS_flag == 1:
            base_result_path += "_WRS"

    if args.KL_Loss == 1:
        base_result_path += "_KL_" + str(args.KL_Loss_weight) + "_T" + str(args.KL_Loss_T)

    base_result_path += "_lr" + str(args.learning_rate) + "_B" + str(args.batch_size)
    base_result_path += "/" + args.target + str(args.time) + "/"
    if not os.path.exists(base_result_path):
        os.makedirs(base_result_path)
    return base_result_path


class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        if args.network == 'resnet18':
            model = resnet18(
                pretrained=True,
                device=device,
                classes=args.n_classes,
                domains=args.n_domains,
                network=args.network,
                domain_discriminator_flag=args.domain_discriminator_flag,
                grl=args.grl,
                lambd=args.lambd,
                drop_percent=args.drop_percent,
                wrs_flag=args.filter_WRS_flag,
                recover_flag=args.recover_flag,
            )
        elif args.network == 'resnet50':
            model = resnet50(
                pretrained=True,
                device=device,
                classes=args.n_classes,
                domains=args.n_domains,
                network=args.network,
                domain_discriminator_flag=args.domain_discriminator_flag,
                grl=args.grl,
                lambd=args.lambd,
                drop_percent=args.drop_percent,
                wrs_flag=args.filter_WRS_flag,
                recover_flag=args.recover_flag,
            )
        else:
            raise NotImplementedError("Not Implemented Network.")

        self.model = model.to(device)
        self.source_loader, self.val_loader = data_helper.get_train_dataloader(args, patches=model.is_patch_based())
        self.target_loader = data_helper.get_val_dataloader(args, patches=model.is_patch_based())
        self.test_loaders = {"val": self.val_loader, "test": self.target_loader}
        self.len_dataloader = len(self.source_loader)
        print("Dataset size: train %d, val %d, test %d" % (len(self.source_loader.dataset),
                                                           len(self.val_loader.dataset),
                                                           len(self.target_loader.dataset)))

        self.optimizer_scatter, self.scheduler_scatter = get_optim_and_scheduler_scatter(model=model,
                                                                                         network=args.network,
                                                                                         epochs=args.epochs,
                                                                                         lr=args.learning_rate,
                                                                                         nesterov=args.nesterov)
        self.n_classes = args.n_classes
        self.base_result_path = get_results_path(args)

        self.val_best = 0.0
        self.test_corresponding = 0.0

        self.criterion = nn.CrossEntropyLoss()
        self.domain_criterion = nn.CrossEntropyLoss()

        self.domain_discriminator_flag = args.domain_discriminator_flag
        self.domain_loss_flag = args.domain_loss_flag
        self.discriminator_layers = args.discriminator_layers

        self.layer_wise_prob = args.layer_wise_prob

    def select_layers(self, layer_wise_prob):
        # layer_wise_prob: prob for layer-wise dropout
        layer_index = np.random.randint(len(self.args.discriminator_layers), size=1)[0]
        layer_select = self.discriminator_layers[layer_index]
        layer_drop_flag = [0, 0, 0, 0]
        if random.random() <= layer_wise_prob:
            layer_drop_flag[layer_select - 1] = 1
        return layer_drop_flag

    def _do_epoch(self, epoch=None):
        self.model.train()

        CE_loss = 0.0
        batch_num = 0.0
        class_right = 0.0
        class_total = 0.0

        CE_domain_loss = [0.0 for i in range(5)]
        domain_right = [0.0 for i in range(5)]
        CE_domain_losses_avg = 0.0
        KL_loss = 0.0

        for it, ((data, class_l, domain_l), d_idx) in enumerate(self.source_loader):
            if self.args.KL_Loss == 1:
                data = torch.cat((data, data)).to(self.device)
                class_l = torch.cat((class_l, class_l)).to(self.device)
                domain_l = torch.cat((domain_l, domain_l)).to(self.device)
            else:
                data = data.to(self.device)
                class_l = class_l.to(self.device)
                domain_l = domain_l.to(self.device)

            layer_drop_flag = self.select_layers(layer_wise_prob=self.layer_wise_prob)
            optimizer = self.optimizer_scatter

            class_logit, domain_logit = self.model(x=data, domain_labels=domain_l, layer_drop_flag=layer_drop_flag)
            class_loss = self.criterion(class_logit, class_l)
            CE_loss += class_loss
            domain_losses_avg = torch.tensor(0.0).to(device=self.device)

            if self.domain_discriminator_flag == 1:
                domain_losses = []
                for i, logit in enumerate(domain_logit):
                    domain_loss = self.domain_criterion(logit, domain_l)
                    domain_losses.append(domain_loss)
                    CE_domain_loss[i] += domain_loss
                domain_losses = torch.stack(domain_losses, dim=0)
                domain_losses_avg = domain_losses.mean(dim=0)
            CE_domain_losses_avg += domain_losses_avg

            loss = 0.0
            loss += class_loss
            if self.domain_loss_flag == 1:
                loss += domain_losses_avg
            if self.args.KL_Loss == 1:
                batch_size = int(class_logit.shape[0] / 2)
                class_logit_1 = class_logit[:batch_size]
                class_logit_2 = class_logit[batch_size:]
                kl_loss = compute_kl_loss(class_logit_1, class_logit_2, T=self.args.KL_Loss_T)
                loss += self.args.KL_Loss_weight * kl_loss
                KL_loss += kl_loss

            for opt in optimizer:
                opt.zero_grad()
            loss.backward()
            for opt in optimizer:
                opt.step()

            _, class_pred = class_logit.max(dim=1)
            class_right_batch = torch.sum(class_pred == class_l.data)
            class_right += class_right_batch

            domain_right_batch = [torch.tensor(0.0).cuda() for i in range(5)]
            if self.domain_discriminator_flag == 1:
                for i, logit in enumerate(domain_logit):
                    _, domain_pred = logit.max(dim=1)
                    domain_right_batch[i] = torch.sum(domain_pred == domain_l.data)
                    domain_right[i] += domain_right_batch[i]
            batch_num += 1

            data_shape = data.shape[0]
            class_total += data_shape

            self.logger.log(it, len(self.source_loader),
                            {
                                "class": class_loss.item(),
                                "domain": domain_losses_avg.item(),
                                "loss": loss.item(),
                            },
                            {
                                "class": class_right_batch,
                            }, data_shape)
        CE_loss = float(CE_loss) / batch_num
        CE_domain_losses_avg = float(CE_domain_losses_avg / batch_num)
        CE_domain_loss = [float(loss / batch_num) for loss in CE_domain_loss]

        class_acc = float(class_right) / class_total
        domain_acc = [float(right / class_total) for right in domain_right]

        KL_loss = float(KL_loss / batch_num)

        result_domain_acc = ", Domain Acc"
        result_domain_loss = ", Domain loss"
        if self.domain_discriminator_flag == 1:
            result_domain_loss += ", Avg: " + str(format(CE_domain_losses_avg, '.4f'))
            for i in range(5):
                result_domain_acc += ", L" + str(i) + ": " + str(format(domain_acc[i], ".4f"))
                result_domain_loss += ", L" + str(i) + ": " + str(format(CE_domain_loss[i], '.4f'))

        result = "train" + ": Epoch: " + str(epoch) \
                 + ", CELoss: " + str(format(CE_loss, '.4f')) \
                 + ", ACC: " + str(format(class_acc, '.4f')) \
                 + result_domain_loss \
                 + result_domain_acc \
                 + ", KL loss: " + str(format(KL_loss, '.4f')) \
                 + '\n'
        with open(self.base_result_path + "/" + "train" + ".txt", "a") as f:
            f.write(result)

        self.model.eval()
        with torch.no_grad():
            val_test_acc = []
            for phase, loader in self.test_loaders.items():
                class_acc, CE_loss = self.do_test(loader)
                val_test_acc.append(class_acc)

                result = phase + ": Epoch: " + str(epoch) \
                         + ", CELoss: " + str(format(CE_loss, '.4f')) \
                         + ", ACC: " + str(format(class_acc, '.4f')) \
                         + "\n"
                with open(self.base_result_path + "/" + phase + ".txt", "a") as f:
                    f.write(result)

                self.logger.log_test(phase, {"class": class_acc})
                self.results[phase][self.current_epoch] = class_acc
            if val_test_acc[0] >= self.val_best:
                self.val_best = val_test_acc[0]
                self.save_model(mode="best")

    def do_eval(self, model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint, strict=False)
        self.model.eval()
        with torch.no_grad():
            for phase, loader in self.test_loaders.items():
                class_acc, CE_loss = self.do_test(loader)
                result = phase + ": CELoss: " + str(format(CE_loss, '.4f')) \
                         + ", ACC: " + str(format(class_acc, '.4f'))
                print(result)

    def save_model(self, mode="best"):
        model_path = self.base_result_path + "models/"
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_name = "model_" + mode + ".pt"
        torch.save(self.model.state_dict(), os.path.join(model_path, model_name))

    def do_test(self, loader):
        class_right = 0.0
        CE_loss = 0.0
        batch_num = 0
        for it, ((data, class_l, domain_l), _) in enumerate(loader):
            data, class_l = data.to(self.device), class_l.to(self.device)
            class_logit, _ = self.model(x=data, layer_drop_flag=[0, 0, 0, 0])
            class_loss = self.criterion(class_logit, class_l)
            _, cls_pred = class_logit.max(dim=1)

            CE_loss += class_loss
            class_right += torch.sum(cls_pred == class_l.data)
            batch_num += 1
        CE_loss = float(CE_loss) / batch_num
        class_acc = float(class_right) / len(loader.dataset)
        return class_acc, CE_loss

    def do_training(self):
        self.logger = Logger(self.args, update_frequency=30)
        self.results = {"val": torch.zeros(self.args.epochs), "test": torch.zeros(self.args.epochs)}
        for self.current_epoch in range(self.args.epochs):
            start_time = time.time()
            self._do_epoch(self.current_epoch)
            self.logger.new_epoch(self.scheduler_scatter[0].get_last_lr())
            for scl in self.scheduler_scatter:
                scl.step()
            end_time = time.time()
            print("Time for one epoch is " + str(format(end_time-start_time, '.0f')) + "s")
        self.save_model(mode="last")
        val_res = self.results["val"]
        test_res = self.results["test"]
        idx_best = val_res.argmax()
        line = "Best val %g, corresponding test %g - best test: %g, best epoch: %g" % (
        val_res.max(), test_res[idx_best], test_res.max(), idx_best)
        print(line)
        with open(self.base_result_path+"test.txt", "a") as f:
            f.write(line+"\n")
        self.logger.save_best(test_res[idx_best], test_res.max())
        return self.logger, self.model


domain_map = {
    'PACS': ['photo', 'art_painting', 'cartoon', 'sketch'],
    'PACS_random_split': ['photo', 'art_painting', 'cartoon', 'sketch'],
    'OfficeHome': ['Art', 'Clipart', 'Product', 'RealWorld'],
    'VLCS': ["CALTECH", "LABELME", "PASCAL", "SUN"],
}

classes_map = {
    'PACS': 7,
    'PACS_random_split': 7,
    'OfficeHome': 65,
    'VLCS': 5,
}

val_size_map = {
    'PACS': 0.1,
    'PACS_random_split': 0.1,
    'OfficeHome': 0.1,
    'VLCS': 0.3,
}

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True

def get_domain(name):
    if name not in domain_map:
        raise ValueError('Name of dataset unknown %s' %name)
    return domain_map[name]

def main():
    args = get_args()

    domain = get_domain(args.data)
    args.target = domain.pop(args.target)
    args.source = domain
    print("Target domain: {}".format(args.target))
    args.data_root = os.path.join(args.data_root, "PACS") if "PACS" in args.data else os.path.join(args.data_root,
                                                                                                   args.data)
    args.n_classes = classes_map[args.data]
    args.n_domains = len(domain)
    args.val_size = val_size_map[args.data]
    setup_seed(args.time)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = Trainer(args, device)
    if args.eval:
        model_path = args.eval_model_path
        trainer.do_eval(model_path=model_path)
        return
    trainer.do_training()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
