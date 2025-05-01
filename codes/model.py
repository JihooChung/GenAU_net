pip install kagglehub==0.3.6
pip install albumentations

import kagglehub
import numpy as np
import pandas as pd

import os
import cv2
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm
import time
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist

BASE_PATH= "/kaggle/input/lgg-mri-segmentation/kaggle_3m"
BASE_LEN = 27
BASE_LEN = 67
END_LEN = 4
END_MASK_LEN = 9
IMG_SIZE = 512

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)

#-----------Data Loading-----------

path = kagglehub.dataset_download("mateuszbuda/lgg-mri-segmentation")

data = []

for dir_ in os.listdir(BASE_PATH):
    dir_path = os.path.join(BASE_PATH, dir_)
    if os.path.isdir(dir_path):
        for filename in os.listdir(dir_path):
            img_path = os.path.join(dir_path, filename)
            data.append([dir_, img_path])

df = pd.DataFrame(data, columns=["dir_name", "image_path"])
df_imgs = df[~df["image_path"].str.contains("mask")]
df_masks = df[df["image_path"].str.contains("mask")]

imgs = sorted(df_imgs["image_path"].values, key= lambda x: x[BASE_LEN: -END_LEN])
masks = sorted(df_masks["image_path"].values, key=lambda x: x[BASE_LEN: -END_MASK_LEN])

gene_df = pd.read_csv('/imputed_genomic_data.csv')
gene_df = gene_df.iloc[:, :8]

df_selected_columns = gene_df.iloc[:, 1:8]
combined_array = df_selected_columns.values.tolist()
gene_df['gene_array'] = combined_array

gene_df = gene_df[['Patient','gene_array']]

dir_names = df_imgs['dir_name']

data = []
for image, mask, dir_name in zip(imgs, masks, dir_names):
    data.append({
        'patient': dir_name,  # dir_name을 patient로 사용
        'image_path': image,
        'mask_path': mask
    })
df_patient_images_masks = pd.DataFrame(data)
dff = df_patient_images_masks
dff['patient_id'] = dff['patient'].str[:12]

#print(dff.shape) -> (3929, 4)

def pos_neg_diagnosis(mask_path):
    val = np.max(cv2.imread(mask_path))
    if val > 0: return 1
    else: return 0

gene_dff = pd.merge(dff, gene_df, left_on='patient_id', right_on='Patient', how='left')
gene_dff = gene_dff[['patient','image_path','mask_path','gene_array']]      
gene_dff["diagnosis"] = gene_dff["mask_path"].apply(lambda x: pos_neg_diagnosis(x))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#-----------Data Preprocessing-----------

class BrainMRIDataset:
    def __init__(self, df, transforms):
        self.df = df
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = cv2.imread(self.df.iloc[idx, 1])
        mask = cv2.imread(self.df.iloc[idx, 2], 0)
        gene_info = self.df.iloc[idx, 3]

        augmented = self.transforms(image=image,
                                   mask=mask)

        image = augmented["image"]
        mask = augmented["mask"]
        if len(mask.shape) == 2:  # H, W
          mask = mask.unsqueeze(0)  # 1, H, W

        return image, mask, gene_info

def custom_collate_fn(batch):
    images, masks, gene_infos = zip(*batch)
    images = torch.stack(images)
    masks = torch.stack(masks)
    return images, masks, list(gene_infos)
  
def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

PATCH_SIZE = 128

transforms = A.Compose([
    A.Resize(width = PATCH_SIZE, height = PATCH_SIZE, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Transpose(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04, rotate_limit=0, p=0.25),
    A.Normalize(p=1.0),
    ToTensorV2(),

])

unique_patients = gene_dff['patient'].unique()
train_patients, val_test_patients = train_test_split(unique_patients, stratify=gene_dff.drop_duplicates('patient')['diagnosis'], test_size=0.2, random_state=SEED)
val_patients, test_patients = train_test_split(val_test_patients, stratify=gene_dff[gene_dff['patient'].isin(val_test_patients)].drop_duplicates('patient')['diagnosis'], test_size=0.5, random_state=SEED)

train_df = gene_dff[gene_dff['patient'].isin(train_patients)].reset_index(drop=True)
val_df = gene_dff[gene_dff['patient'].isin(val_patients)].reset_index(drop=True)
test_df = gene_dff[gene_dff['patient'].isin(test_patients)].reset_index(drop=True)

#print(f"Train: {train_df.shape} Val: {val_df.shape} Test: {test_df.shape}")
# -> Train: (3092, 5) Val: (431, 5) Test: (406, 5)

g = torch.Generator()
g.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

train_dataset = BrainMRIDataset(train_df, transforms=transforms)
train_dataloader = DataLoader(train_dataset, batch_size=26, num_workers=2, shuffle=True, collate_fn=custom_collate_fn, worker_init_fn=seed_worker, generator=g)

val_dataset = BrainMRIDataset(val_df, transforms=transforms)
val_dataloader = DataLoader(val_dataset, batch_size=26, num_workers=2, shuffle=True, collate_fn=custom_collate_fn, worker_init_fn=seed_worker, generator=g)

test_dataset = BrainMRIDataset(test_df, transforms=transforms)
test_dataloader = DataLoader(test_dataset, batch_size=26, num_workers=2, shuffle=False, collate_fn=custom_collate_fn, worker_init_fn=seed_worker, generator=g)

#-----------Model Architecture-----------
class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv = nn.Sequential(
                                  nn.Conv2d(ch_in, ch_out,
                                            kernel_size=3, stride=1,
                                            padding=1, bias=True),
                                  nn.BatchNorm2d(ch_out),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(ch_out, ch_out,
                                            kernel_size=3, stride=1,
                                            padding=1, bias=True),
                                  nn.BatchNorm2d(ch_out),
                                  nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class UpConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.up = nn.Sequential(
                                nn.Upsample(scale_factor=2),
                                nn.Conv2d(ch_in, ch_out,
                                         kernel_size=3,stride=1,
                                         padding=1, bias=True),
                                nn.BatchNorm2d(ch_out),
                                nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = x = self.up(x)
        return x

class GABlock(nn.Module):
    def __init__(self, f_g, f_l, f_int, gene_info_size=7):
        super().__init__()

        self.w_g = nn.Sequential(
                                nn.Conv2d(f_g, f_int,
                                         kernel_size=1, stride=1,
                                         padding=0, bias=True),
                                nn.BatchNorm2d(f_int)
        )

        self.w_x = nn.Sequential(
                                nn.Conv2d(f_l, f_int,
                                         kernel_size=1, stride=1,
                                         padding=0, bias=True),
                                nn.BatchNorm2d(f_int)
        )

        self.gene = nn.Sequential(
            nn.Linear(gene_info_size, f_int),
            nn.Sigmoid()
        )

        self.psi = nn.Sequential(
                                nn.Conv2d(f_int, 1,
                                         kernel_size=1, stride=1,
                                         padding=0,  bias=True),
                                nn.BatchNorm2d(1),
                                nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x, gene_info):
        g1 = self.w_g(g)
        x1 = self.w_x(x)

        gene_info = self.gene(gene_info)
        gene_info = gene_info.view(-1, gene_info.size(1), 1, 1)

        psi = self.relu((g1+x1)+gene_info)
        psi = self.psi(psi)

        return psi*x

class GenAU_net(nn.Module):
    def __init__(self, n_classes=1, in_channel=3, out_channel=1):
        super().__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = ConvBlock(ch_in=in_channel, ch_out=64)
        self.conv2 = ConvBlock(ch_in=64, ch_out=128)
        self.conv3 = ConvBlock(ch_in=128, ch_out=256)
        self.conv4 = ConvBlock(ch_in=256, ch_out=512)
        self.conv5 = ConvBlock(ch_in=512, ch_out=1024)

        self.up5 = UpConvBlock(ch_in=1024, ch_out=512)
        self.att5 = GABlock(f_g=512, f_l=512, f_int=256)
        self.upconv5 = ConvBlock(ch_in=1024, ch_out=512)

        self.up4 = UpConvBlock(ch_in=512, ch_out=256)
        self.att4 = GABlock(f_g=256, f_l=256, f_int=128)
        self.upconv4 = ConvBlock(ch_in=512, ch_out=256)

        self.up3 = UpConvBlock(ch_in=256, ch_out=128)
        self.att3 = GABlock(f_g=128, f_l=128, f_int=64)
        self.upconv3 = ConvBlock(ch_in=256, ch_out=128)

        self.up2 = UpConvBlock(ch_in=128, ch_out=64)
        self.att2 = GABlock(f_g=64, f_l=64, f_int=32)
        self.upconv2 = ConvBlock(ch_in=128, ch_out=64)

        self.conv_1x1 = nn.Conv2d(64, out_channel,
                                  kernel_size=1, stride=1, padding=0)
    def forward(self, x, gene_info):
        # encoder
        x1 = self.conv1(x)

        x2 = self.maxpool(x1)
        x2 = self.conv2(x2)

        x3 = self.maxpool(x2)
        x3 = self.conv3(x3)

        x4 = self.maxpool(x3)
        x4 = self.conv4(x4)

        x5 = self.maxpool(x4)
        x5 = self.conv5(x5)

        # decoder + concat
        d5 = self.up5(x5)
        x4 = self.att5(g=d5, x=x4, gene_info=gene_info)
        d5 = torch.concat((x4, d5), dim=1)
        d5 = self.upconv5(d5)

        d4 = self.up4(d5)
        x3 = self.att4(g=d4, x=x3, gene_info=gene_info)
        d4 = torch.concat((x3, d4), dim=1)
        d4 = self.upconv4(d4)

        d3 = self.up3(d4)
        x2 = self.att3(g=d3, x=x2, gene_info=gene_info)
        d3 = torch.concat((x2, d3), dim=1)
        d3 = self.upconv3(d3)

        d2 = self.up2(d3)
        x1 = self.att2(g=d2, x=x1, gene_info=gene_info)
        d2 = torch.concat((x1, d2), dim=1)
        d2 = self.upconv2(d2)

        d1 = self.conv_1x1(d2)

        return d1

genaunet = GenAU_net(n_classes=1).to(device)
opt = torch.optim.Adamax(genaunet.parameters(), lr=1e-3)

#-----------Model Training-----------

def dice_coef_metric(inputs, target):

    if inputs.ndim == 4:
        inputs = inputs.squeeze(1)
    if target.ndim == 4:
        target = target.squeeze(1)

    intersection = (inputs * target).sum(axis=(1, 2))  # per image
    union = inputs.sum(axis=(1, 2)) + target.sum(axis=(1, 2))  # per image

    dice = (2.0 * intersection + 1e-7) / (union + 1e-7)  # avoid div0
    return dice.mean()

def compute_iou(model, loader, threshold=0.3):
    valloss = 0

    with torch.no_grad():

        for i_step, (data, target, gene_info) in enumerate(loader):

            data = data.to(device)
            target = target.to(device)
            target = (target > 0).float()
            gene_info = torch.tensor(gene_info, dtype=torch.float32).to(device)

            outputs = model(data, gene_info)

            out_cut = np.copy(outputs.data.cpu().numpy())
            out_cut[np.nonzero(out_cut < threshold)] = 0.0
            out_cut[np.nonzero(out_cut >= threshold)] = 1.0
            picloss = dice_coef_metric(out_cut, target.cpu().numpy())
            valloss += picloss

    return valloss / len(loader)

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice

def train_model(model_name, model, train_loader, val_loader, train_loss, optimizer, lr_scheduler, num_epochs):
    print(f"[INFO] Model is initializing... {model_name}")

    loss_history = []
    train_history = []
    val_history = []

    for epoch in range(num_epochs):
        model.train()

        losses = []
        train_iou = []

        for i_step, (data, target, gene_info) in enumerate(tqdm(train_loader)):
            data = data.to(device)
            target = target.to(device)
            target = (target > 0).float()

            gene_info = np.array(gene_info)
            gene_info = torch.tensor(gene_info, dtype=torch.float32).to(device)

            outputs = model(data, gene_info)

            out_cut = np.copy(outputs.data.cpu().numpy())
            out_cut[np.nonzero(out_cut < 0.5)] = 0.0
            out_cut[np.nonzero(out_cut >= 0.5)] = 1.0

            train_dice = dice_coef_metric(out_cut, target.cpu().numpy())

            loss = train_loss(outputs, target)

            losses.append(loss.item())
            train_iou.append(train_dice)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_mean_iou = compute_iou(model, val_loader)

        loss_history.append(np.array(losses).mean())
        train_history.append(np.array(train_iou).mean())
        val_history.append(val_mean_iou)

        print("Epoch [%d]" % (epoch))
        print("\nMean DICE on train:", np.array(train_iou).mean(),
              "\nMean DICE on validation:", val_mean_iou)

    return loss_history, train_history, val_history

%%time
num_ep = 50

lh, th, vh = train_model("Attention UNet", genaunet, train_dataloader, val_dataloader, DiceLoss(), opt, False, num_ep)

#-----------Model Evaluation-----------

def dice_coef_metric(inputs, target):
    if inputs.ndim == 4:
        inputs = inputs.squeeze(1)
    if target.ndim == 4:
        target = target.squeeze(1)

    intersection = (inputs * target).sum(axis=(1, 2))  # per image
    union = inputs.sum(axis=(1, 2)) + target.sum(axis=(1, 2))  # per image

    dice = (2.0 * intersection + 1e-7) / (union + 1e-7)  # avoid div0
    return dice.mean()
  
def f1_score(inputs, target, epsilon=1e-6):
    if inputs.ndim == 4:
        inputs = inputs.squeeze(1)
    if target.ndim == 4:
        target = target.squeeze(1)

    tp = (inputs * target).sum(axis=(1, 2))
    fp = ((1 - target) * inputs).sum(axis=(1, 2))
    fn = (target * (1 - inputs)).sum(axis=(1, 2))

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)

    no_pos_pred = inputs.sum(axis=(1, 2)) == 0
    no_pos_true = target.sum(axis=(1, 2)) == 0
    both_empty = no_pos_pred & no_pos_true

    f1[both_empty] = 1.0
    return f1.mean()

def mean_iou(inputs, target, num_classes=2):
    ious = []
    for i in range(num_classes):
        tp = np.logical_and(inputs == i, target == i).sum().astype(float)
        fp = np.logical_and(inputs == i, target != i).sum().astype(float)
        fn = np.logical_and(inputs != i, target == i).sum().astype(float)
        iou = tp / (tp + fp + fn + 1e-6)
        ious.append(iou)

    return np.mean(ious)

def assd(inputs, target):

    output_boundary = np.argwhere(inputs > 0)
    target_boundary = np.argwhere(target > 0)

    if len(output_boundary) == 0 or len(target_boundary) == 0:
        return float('nan')  # 경계가 없으면 nan 반환

    # 두 경계들 사이의 대칭적인 평균 경계 거리 계산
    distances1 = cdist(output_boundary, target_boundary, metric='euclidean')
    distances2 = cdist(target_boundary, output_boundary, metric='euclidean')


    assd = np.mean(np.min(distances1, axis=1)) + np.mean(np.min(distances2, axis=1))
    assd /= 2.0

    return assd

def compute_metrics(model, loader, threshold=0.3, num_classes=2):
    model.eval()
    all_dice_scores = []
    all_miou_scores = []
    all_assd_scores = []

    with torch.no_grad():
        for i_step, (data, target, gene_info) in enumerate(loader):
            data = data.to(device)
            target = target.to(device)
            target = (target > 0).float()

            gene_info = np.array(gene_info)
            gene_info = torch.tensor(gene_info, dtype=torch.float32).to(device)

            outputs = model(data, gene_info)

            # Apply thresholding
            out_cut = np.copy(outputs.data.cpu().numpy())
            out_cut[np.nonzero(out_cut < threshold)] = 0.0
            out_cut[np.nonzero(out_cut >= threshold)] = 1.0

            # Compute metrics
            dice_score = dice_coef_metric(out_cut, target.cpu().numpy())
            miou = mean_iou(out_cut, target.cpu().numpy(), num_classes=num_classes)
            assd_ = assd(out_cut, target.cpu().numpy())

            # Append to lists
            all_dice_scores.append(dice_score)
            all_miou_scores.append(miou)
            all_assd_scores.append(assd_)

    # Compute mean and std
    return {
        "Dice": (np.mean(all_dice_scores), np.std(all_dice_scores)),
        "mIoU": (np.mean(all_miou_scores), np.std(all_miou_scores)),
        "ASSD": (np.nanmean(all_assd_scores), np.nanstd(all_assd_scores))
    }

print(compute_metrics(genaunet, test_dataloader, threshold=0.3, num_classes=2))

def visualize_segmentation(model, loader, num_samples=5, threshold=0.3):
    model.eval()
    sample_count = 0
    fig, axes = plt.subplots(num_samples, 3, figsize=(10, num_samples * 3))

    with torch.no_grad():
        for i, (data, target, gene_info) in enumerate(loader):
            if sample_count >= num_samples:
                break

            data = data.to(device)
            target = target.to(device)
            target = (target > 0).float()

            gene_info = np.array(gene_info)
            gene_info = torch.tensor(gene_info, dtype=torch.float32).to(device)

            for j in range(data.size(0)):
                if sample_count >= num_samples:
                    break

                if not(np.any(target.cpu().numpy()[j, 0] > 0)):
                    continue
                
                outputs = model(data, gene_info)
                outputs = outputs.data.cpu().numpy()

                out_cut = np.copy(outputs)
                out_cut[np.nonzero(out_cut < threshold)] = 0.0
                out_cut[np.nonzero(out_cut >= threshold)] = 1.0

                img = data.cpu().numpy().transpose(0, 2, 3, 1)[j]  # (C, H, W) -> (H, W, C)

                mask_gt = target.cpu().numpy()[j, 0]
                mask_pred = out_cut[j, 0]
              
                img = np.mean(img, axis=-1)
                img = (img - img.min()) / (img.max() - img.min())
                img = (img * 255).astype(np.uint8)
                if img.ndim == 2:
                    img = np.stack([img] * 3, axis=-1)
                mask_gt_color = np.zeros((*mask_gt.shape, 3), dtype=np.uint8)
                mask_gt_color[mask_gt == 1] = [255, 0, 0]
                mask_pred_color = np.zeros((*mask_pred.shape, 3), dtype=np.uint8)
                mask_pred_color[mask_pred == 1] = [0, 0, 255]

                overlay_gt = cv2.addWeighted(img, 0.7, mask_gt_color, 0.3, 0)
                overlay_pred = cv2.addWeighted(img, 0.7, mask_pred_color, 0.6, 0)

                axes[sample_count, 0].imshow(img)
                axes[sample_count, 0].set_title("Original Image")
                axes[sample_count, 0].axis("off")

                axes[sample_count, 1].imshow(overlay_gt)
                axes[sample_count, 1].set_title("GroundTruth Overlay")
                axes[sample_count, 1].axis("off")

                axes[sample_count, 2].imshow(overlay_pred)
                axes[sample_count, 2].set_title("Prediction Overlay")
                axes[sample_count, 2].axis("off")

                sample_count += 1

    plt.tight_layout()
    plt.show()

visualize_segmentation(genaunet, test_dataloader, num_samples=10, threshold=0.3)
