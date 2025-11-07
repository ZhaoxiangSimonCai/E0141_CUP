from random import random

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    classification_report,
    roc_auc_score,
)
import numpy as np  # linear algebra
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import sys
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import torch.nn as nn
from tqdm import trange, tqdm
import time
from PIL import ImageFile
import torchvision
from sklearn.model_selection import StratifiedKFold
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True
import warnings

device = "cuda"
seed = 42
BATCH_SIZE = 512
NUM_WORKERS = 1
LEARNING_RATE = 1e-4
LR_STEP = 6
LR_FACTOR = 0.1

LOG_FREQ = 50
WD = 0.0001


class SingleSlideDataset(Dataset):
    def __init__(self, patch_list, mode, label):
        assert mode in ["train", "val", "test"]

        self.patch_list = patch_list
        self.mode = mode
        self.label = label

        transforms_list = []
        self.transforms_pre = transforms.Compose(transforms_list)

        if self.mode == "train":
            transforms_list.extend(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomChoice(
                        [
                            transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
                            transforms.RandomAffine(
                                degrees=(0, 360), translate=(0.1, 0.1), scale=(0.8, 1.2)
                            ),
                        ]
                    ),
                ]
            )

        transforms_list.extend([transforms.ToTensor()])
        self.transforms = transforms.Compose(transforms_list)
        self.to_tensor = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        """Returns: tuple (sample, target)"""
        patch = self.patch_list[index]
        sample = Image.fromarray(patch)
        if sample.mode != "RGB":
            sample = sample.convert("RGB")
        image = self.transforms(sample)
        return image, self.label

    def __len__(self):
        return len(self.patch_list)


class RandomPatchDataset(Dataset):
    def __init__(self, patch_list, labels):
        self.patch_list = patch_list
        self.labels = labels

    def __getitem__(self, index):
        """Returns: tuple (sample, target)"""
        patch = self.patch_list[index]
        label = self.labels[index]

        return patch, label

    def __len__(self):
        return len(self.patch_list)


class OmicsDataset(Dataset):
    def __init__(self, omics_df, labels):
        self.omics_df = omics_df
        self.labels = labels

    def __getitem__(self, index):
        """Returns: tuple (sample, target)"""
        omics_data = self.omics_df.iloc[index, :]
        label = self.labels[index]

        return torch.FloatTensor(omics_data), label

    def __len__(self):
        return self.omics_df.shape[0]


class ImageProteomicsDataset(Dataset):
    def __init__(self, img_dataset, omics_dataset):
        self.img_dataset = img_dataset
        self.omics_dataset = omics_dataset

    def __getitem__(self, index):
        """Returns: tuple (sample, target)"""
        img_data, img_label = self.img_dataset[index]
        omics_data, omics_label = self.omics_dataset[index]

        return (img_data, omics_data), img_label

    def __len__(self):
        return len(self.omics_dataset)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ImageProteomicsE2EModel(nn.Module):
    def __init__(self, img_model, omic_model, num_classes):
        super(ImageProteomicsE2EModel, self).__init__()
        self.num_classes = num_classes
        self.img_model = img_model
        self.omic_model = omic_model
        self.total_dim = self.img_model.fc.in_features + self.omic_model.fc2.in_features
        self.img_model.fc = nn.Identity()
        self.omic_model.fc2 = nn.Identity()
        self.fc = nn.Linear(self.total_dim, num_classes)

    def forward(self, img_x, omics_x):
        img_x = self.img_model(img_x)
        omics_x = self.omic_model(omics_x)
        x = torch.cat([img_x, omics_x], dim=1)
        x = self.fc(x)
        return x


def get_patches_DA(df, label_to_id, target, model, k="auto", mode="max"):
    res_patches = []
    res_labels = []
    for index, row in tqdm(df.iterrows(), total=len(df)):
        img_lims = row["Image LIMS ID"]
        label = row[target]
        label_id = label_to_id[label]
        if not os.path.exists(f"../data/patches_256window/{img_lims}.npy"):
            continue
        patches = np.load(f"../data/patches_256window/{img_lims}.npy")
        patches = patches[
            np.random.choice(range(0, patches.shape[0]), BATCH_SIZE * 1), :, :, :
        ]
        img_dataset = SingleSlideDataset(patches, mode="train", label=label)
        img_loader = DataLoader(
            img_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
        )

        patch_out = []
        patches_tensor = []
        if mode == "random":
            with torch.no_grad():
                for i, (input_, targets) in enumerate(img_loader):
                    patches_tensor.extend(input_)
            res_patches.extend(patches_tensor)
            res_labels.extend([label_id] * len(patches_tensor))
        elif mode == "max":
            with torch.no_grad():
                for i, (input_, targets) in enumerate(img_loader):
                    patches_tensor.extend(input_)
                    output = model(input_.to(device))  # N x C
                    output = output[:, label_id]
                    patch_out.extend(output)
            if len(patch_out) == 0:
                continue
            if k == "auto":
                k = BATCH_SIZE // 4
            elif k == "all":
                k = len(patch_out)
            patches_tensor = torch.stack(patches_tensor)
            _, max_idx = torch.topk(torch.stack(patch_out), k)
            res_patches.extend(patches_tensor[max_idx.cpu().numpy()])
            res_labels.extend([label_id] * k)
        else:
            raise ValueError("mode must be either 'random' or 'max'")
    return res_patches, res_labels


def get_random_patches(df, label_to_id, target, k="all", mode="val"):
    res_patches = []
    res_labels = []
    res_samples = []
    for index, row in tqdm(df.iterrows(), total=len(df)):
        img_lims = row["Image LIMS ID"]
        sample_id = row["LIMS-ID1"]
        label = row[target]
        label_id = label_to_id[label]
        if not os.path.exists(f"../data/patches_256window/{img_lims}.npy"):
            continue
        patches = np.load(f"../data/patches_256window/{img_lims}.npy")
        if k == "all":
            k = len(patches)
        k = min(k, len(patches))
        patches = patches[np.random.choice(range(0, patches.shape[0]), k), :, :, :]
        patches_tensor = torch.from_numpy(np.moveaxis(patches, -1, 1))
        if mode == "val":
            res_patches.append(patches_tensor)
            res_labels.append(label_id)
            res_samples.append(sample_id)
        elif mode == "train":
            res_patches.extend(patches_tensor)
            res_labels.extend([label_id] * k)
            res_samples.extend([sample_id] * k)

        else:
            raise ValueError("mode must be either 'train' or 'val'")

    res_samples = pd.DataFrame({"LIMS-ID1": res_samples})
    return res_patches, res_labels, res_samples


def train(train_loader, model, criterion, optimizer, epoch, logging=True):
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_f1 = AverageMeter()
    avg_acc = AverageMeter()
    # avg_auc = AverageMeter()

    model.train()

    end = time.time()
    lr_str = ""

    for i, (input_, targets) in enumerate(train_loader):
        num_steps = len(train_loader)
        if type(input_) == list:
            img_x = input_[0]
            omics_x = input_[1]
            output = model(img_x.float().to(device), omics_x.float().to(device))
        else:
            output = model(input_.float().to(device))
        loss = criterion(output, targets.to(device))

        losses.update(loss.data.item(), output.shape[0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicts = torch.max(output.data, 1)

        predicts = predicts.cpu().numpy()
        targets = targets.cpu().numpy()

        avg_f1.update(f1_score(targets, predicts, average="macro"))
        avg_acc.update(accuracy_score(targets, predicts))

        batch_time.update(time.time() - end)
        end = time.time()

        if logging and i % LOG_FREQ == 0:
            print(
                f"{epoch} [{i}/{num_steps}]\t"
                f"time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                f"loss {losses.val:.4f} ({losses.avg:.4f})\t"
                f"F1 {avg_f1.val:.4f} ({avg_f1.avg:.4f})\t"
                f"accuracy {avg_acc.val:.4f} ({avg_acc.avg:.4f})\t"
                # f'ROCAUC {avg_auc.val:.4f} ({avg_auc.avg:.4f})\t'
                + lr_str
            )
            sys.stdout.flush()

    print(f" * average F1 on train {avg_f1.avg:.4f}")
    print(f" * average Accuracy on train {avg_acc.avg:.4f}")
    # print(f' * average AUC on train {avg_auc.avg:.4f}')
    return avg_acc.avg


def inference(data_loader, model):
    """Returns predictions and targets, if any."""
    model.eval()

    all_predicts, all_confs, all_targets = [], [], []
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            input_, target = data

            output = model(input_.to(device))
            all_confs.append(output)
            _, predicts = torch.max(output.data, 1)
            all_predicts.append(predicts)

            if target is not None:
                all_targets.append(target)

    predicts = torch.cat(all_predicts)
    confs = torch.cat(all_confs)
    targets = torch.cat(all_targets) if len(all_targets) else None

    return predicts, confs, targets


def validate(val_loader, model):
    predicts, confs, targets = inference(val_loader, model)
    predicts = predicts.cpu().numpy()
    # confs = torch.from_numpy(confs)

    targets = targets.cpu().numpy()

    f1 = f1_score(targets, predicts, average="macro")
    acc = accuracy_score(targets, predicts)

    if confs.shape[1] == 2:
        confs = torch.sigmoid(confs).cpu().numpy()
        confs = confs[:, 1]
        # auc = roc_auc_score(targets, confs)
    else:
        confs = torch.softmax(confs, dim=-1).detach().cpu().numpy()
        # auc = roc_auc_score(targets, confs, multi_class='ovo')

    # if not (targets == 1).all() or (targets == 0).all():
    #     auc = roc_auc_score(targets, confs[:, 1], average='micro')

    print(f"val f1: {f1:.4f}")
    print(f"val accuracy: {acc:.4f}")
    # print(f"val AUC: {auc:.4f}")
    # print(classification_report(targets, predicts, target_names=target_names))
    # print(f"val auc {auc:.4f}")

    return acc


def inference_all_patches_single_slide(data_loader, model, patch_batch=512):
    """Returns predictions and targets, if any."""
    model.eval()

    all_predicts, all_confs, all_targets = [], [], []
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            input_, target = data
            img_confs = []

            if type(input_) == list:
                omics_x = input_[1]
                input_ = input_[0]

                # input_: (batch_size, patch_num, 3, w, h)
                for j in range(0, input_.shape[1], patch_batch):
                    tmp_input = input_[0, j : j + patch_batch]
                    output = model(
                        tmp_input.float().to(device),
                        omics_x.repeat(tmp_input.shape[0], 1).float().to(device),
                    )
                    img_confs.append(output)
            else:
                for j in range(0, input_.shape[1], patch_batch):
                    tmp_input = input_[0, j : j + patch_batch]
                    output = model(tmp_input.float().to(device))
                    img_confs.append(output)
            img_avg_confs = torch.mean(torch.cat(img_confs), axis=0)

            all_confs.append(img_avg_confs)
            predicts = torch.argmax(img_avg_confs)
            all_predicts.append(predicts)

            if target is not None:
                all_targets.append(target)

    predicts = torch.stack(all_predicts)
    confs = torch.stack(all_confs)
    targets = torch.cat(all_targets) if len(all_targets) else None

    return predicts, confs, targets


def validate_all_patches_single_slide(val_loader, model):
    predicts, confs, targets = inference_all_patches_single_slide(val_loader, model)
    predicts = predicts.cpu().numpy()
    # confs = torch.from_numpy(confs)

    targets = targets.cpu().numpy()

    f1 = f1_score(targets, predicts, average="macro")
    acc = accuracy_score(targets, predicts)

    if confs.shape[1] == 2:
        confs = torch.sigmoid(confs).cpu().numpy()
        confs = confs[:, 1]
        # auc = roc_auc_score(targets, confs)
    else:
        confs = torch.softmax(confs, dim=-1).detach().cpu().numpy()
        # auc = roc_auc_score(targets, confs, multi_class='ovo')

    # if not (targets == 1).all() or (targets == 0).all():
    #     auc = roc_auc_score(targets, confs[:, 1], average='micro')

    print(f"val f1: {f1:.4f}")
    print(f"val accuracy: {acc:.4f}")
    # print(f"val AUC: {auc:.4f}")
    # print(classification_report(targets, predicts, target_names=target_names))
    # print(f"val auc {auc:.4f}")

    return acc
