#See required.py to see all the libraries used in this code
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
        return float('nan') 

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
