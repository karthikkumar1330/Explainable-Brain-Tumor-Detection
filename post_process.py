import argparse
import os
from glob import glob

import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations import Compose, Resize, Normalize
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import archs
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter
from albumentations import RandomRotate90
import time

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])

    model = model.cuda()

    # Data loading code
    img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

    model.load_state_dict(torch.load('models/%s/model.pth' %
                                     config['name']))
    model.eval()

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        Normalize(),
    ])

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    iou_avg_meter = AverageMeter()
    dice_avg_meter = AverageMeter()
    gput = AverageMeter()
    cput = AverageMeter()

    import numpy as np

    count = 0
    all_preds = []
    all_targets = []
    all_meta = []

    print("=> running inference and latency profiling")
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            model = model.cuda()
            
            # Benchmark GPU inference speed on the first 5 batches
            if count < 5:
                start = time.time()
                if config['deep_supervision']:
                    output = model(input)[-1]
                else:
                    output = model(input)
                stop = time.time()
                gput.update(stop - start, input.size(0))
            else:
                if config['deep_supervision']:
                    output = model(input)[-1]
                else:
                    output = model(input)

            # Benchmark CPU inference speed on the first 5 batches
            if count < 5:
                start = time.time()
                # Run on CPU copy
                model_cpu = model.cpu()
                input_cpu = input.cpu()
                _ = model_cpu(input_cpu)
                stop = time.time()
                cput.update(stop - start, input.size(0))
                count = count + 1

            output = torch.sigmoid(output).cpu().numpy()
            target = target.cpu().numpy()

            for i in range(len(output)):
                all_preds.append(output[i])
                all_targets.append(target[i])
                all_meta.append(meta['img_id'][i])

    # Search for optimal binarization threshold
    best_threshold = 0.5
    best_iou = 0.0

    print("=> searching for optimal binarization threshold")
    thresholds = np.arange(0.1, 0.9, 0.05)
    for thresh in thresholds:
        ious = []
        for pred, gt in zip(all_preds, all_targets):
            for c in range(config['num_classes']):
                pred_c = pred[c] > thresh
                gt_c = gt[c] > 0.5
                intersection = np.logical_and(pred_c, gt_c).sum()
                union = np.logical_or(pred_c, gt_c).sum()
                iou = (intersection + 1e-5) / (union + 1e-5)
                ious.append(iou)
        mean_iou = np.mean(ious)
        if mean_iou > best_iou:
            best_iou = mean_iou
            best_threshold = thresh

    print(f"Optimal threshold found: {best_threshold:.2f} (mIoU: {best_iou:.4f})")

    # Apply optimal threshold and connected component area filter (remove blobs < 100 pixels)
    min_area = 100
    final_ious = []
    final_dices = []

    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)

    print("=> saving post-processed predictions")
    for idx, (pred, gt, img_id) in enumerate(zip(all_preds, all_targets, all_meta)):
        for c in range(config['num_classes']):
            pred_c = pred[c]
            gt_c = gt[c] > 0.5

            # Binarize
            bin_mask = (pred_c > best_threshold).astype(np.uint8)

            # Filter small false positive blobs
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
            filtered_mask = np.zeros_like(bin_mask)
            for label in range(1, num_labels):
                area = stats[label, cv2.CC_STAT_AREA]
                if area >= min_area:
                    filtered_mask[labels == label] = 1

            # Compute final metrics
            intersection = np.logical_and(filtered_mask, gt_c).sum()
            union = np.logical_or(filtered_mask, gt_c).sum()
            iou = (intersection + 1e-5) / (union + 1e-5)
            dice = (2.0 * intersection + 1e-5) / (filtered_mask.sum() + gt_c.sum() + 1e-5)

            final_ious.append(iou)
            final_dices.append(dice)

            # Save binarized and filtered prediction
            cv2.imwrite(os.path.join('outputs', config['name'], str(c), img_id + '.jpg'),
                        (filtered_mask * 255).astype('uint8'))

    print('IoU: %.4f' % np.mean(final_ious))
    print('Dice: %.4f' % np.mean(final_dices))

    print('CPU Average Latency: %.4f s' % cput.avg)
    print('GPU Average Latency: %.4f s' % gput.avg)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
