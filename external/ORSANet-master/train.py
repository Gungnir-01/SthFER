import warnings

warnings.filterwarnings("ignore")
import torch.utils.data as data
from torchvision import transforms

import argparse
import torchvision.datasets as datasets
from sklearn.metrics import f1_score
from time import time
from utils import *
from models.net import ORSA
from torchsampler import ImbalancedDatasetSampler
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='fer2013', help='dataset [fer2013, rafdb, affectnet7, affectnet8, occlu-FER]')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--val_batch_size', type=int, default=64, help='Batch size for validation.')
    parser.add_argument('--modeltype', type=str, default='small', help='small or base or large')
    parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer, adam or sgd.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate for sgd.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for sgd')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=30, help='Total training epochs.')
    parser.add_argument('--gpu', type=str, default='1', help='assign multi-gpus by comma concat')
    parser.add_argument('--use_drae', type=int, default=5, help='use DRAEloss or not')
    parser.add_argument('--weight_drae', type=int, default=0.1, help='weight of DRAEloss')
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume training')
    return parser.parse_args()


def run_training():
    args = parse_args()
    torch.manual_seed(123)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    print("Work on GPU: ", os.environ['CUDA_VISIBLE_DEVICES'])

    data_transforms_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02, 0.1)),
    ])

    data_transforms_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    if args.dataset == "fer2013":
        datapath = '../../FER2013_processed/'
        traindir = os.path.join(datapath, 'train')
        valdir = os.path.join(datapath, 'val')
        num_classes = 7
        train_dataset = datasets.ImageFolder(traindir, transform=data_transforms_train)
        val_dataset = datasets.ImageFolder(valdir, transform=data_transforms_val)
        model = ORSA(img_size=224, num_classes=num_classes, type=args.modeltype)

    elif args.dataset == "rafdb":
        datapath = './data/raf/'
        traindir = os.path.join(datapath, 'train')
        valdir = os.path.join(datapath, 'valid')
        num_classes = 7
        train_dataset = datasets.ImageFolder(traindir, transform=data_transforms_train)
        val_dataset = datasets.ImageFolder(valdir, transform=data_transforms_val)
        model = ORSA(img_size=224, num_classes=num_classes, type=args.modeltype)

    elif args.dataset == "affectnet7":
        datapath = './data/AffectNet_7/'
        num_classes = 7
        traindir = os.path.join(datapath, 'train')
        valdir = os.path.join(datapath, 'valid')
        train_dataset = datasets.ImageFolder(traindir, transform=data_transforms_train)
        val_dataset = datasets.ImageFolder(valdir, transform=data_transforms_val)
        model = ORSA(img_size=224, num_classes=num_classes, type=args.modeltype)

    elif args.dataset == "affectnet8":
        datapath = './data/AffectNet_8/'
        num_classes = 8
        traindir = os.path.join(datapath, 'train')
        valdir = os.path.join(datapath, 'valid')
        train_dataset = datasets.ImageFolder(traindir, transform=data_transforms_train)
        val_dataset = datasets.ImageFolder(valdir, transform=data_transforms_val)
        model = ORSA(img_size=224, num_classes=num_classes, type=args.modeltype)

    elif args.dataset == "occlu-FER":
        datapath = './data/Occlu-FER/'
        num_classes = 8
        traindir = os.path.join(datapath, 'train')
        valdir = os.path.join(datapath, 'valid')
        train_dataset = datasets.ImageFolder(traindir, transform=data_transforms_train)
        val_dataset = datasets.ImageFolder(valdir, transform=data_transforms_val)
        model = ORSA(img_size=224, num_classes=num_classes, type=args.modeltype)

    else:
        return print('dataset name is not correct')

    print('Train set size:', train_dataset.__len__())
    print('Validation set size:', val_dataset.__len__())
    
    if args.dataset == 'raf' or args.dataset == 'fer2013':
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=True,
                                               pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                               sampler=ImbalancedDatasetSampler(train_dataset),
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=False,
                                               pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=args.val_batch_size,
                                            num_workers=args.workers,
                                            shuffle=False,
                                            pin_memory=True)

    print("batch_size:", args.batch_size)

    if args.optimizer == 'adamw':
        base_optimizer = torch.optim.AdamW
    elif args.optimizer == 'adam':
        base_optimizer = torch.optim.Adam
    elif args.optimizer == 'sgd':
        base_optimizer = torch.optim.SGD
    else:
        raise ValueError("Optimizer not supported.")

    optimizer = base_optimizer(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    scaler = torch.amp.GradScaler('cuda')

    model = torch.nn.DataParallel(model)
    model = model.cuda()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Total Parameters: %.3fM' % parameters)

    start_epoch = 1
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['iter'] + 1
            best_acc = checkpoint.get('best_acc', 0.0)
            best_score = checkpoint.get('best_score', 0.0)
            print(f"Resumed at epoch {start_epoch}, best_acc: {best_acc:.4f}, best_score: {best_score:.4f}")
        else:
            print(f"Warning: checkpoint {args.resume} not found, starting from scratch.")

    if not args.resume or not os.path.isfile(args.resume):
        best_acc = 0
        best_score = 0

    CE_criterion = torch.nn.CrossEntropyLoss()
    lsce_criterion = LabelSmoothingCrossEntropy(smoothing=0.2)
    DRAELoss_criterion = DRAELoss()

    best_acc = 0
    best_score = 0
    for i in range(start_epoch, args.epochs + 1):
        train_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        start_time = time()
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {i}/{args.epochs} [Train]', ncols=100)
        for batch_i, (imgs, targets) in enumerate(pbar):
            iter_cnt += 1
            optimizer.zero_grad()
            imgs = imgs.cuda()
            targets = targets.cuda()

            with torch.amp.autocast('cuda'):
                outputs, features = model(imgs)
                CE_loss = CE_criterion(outputs, targets)
                lsce_loss = lsce_criterion(outputs, targets)
                DRAE_loss = DRAELoss_criterion(outputs, targets)
                
                loss = CE_loss + 2 * lsce_loss
                if i > args.use_drae:
                    loss = loss + DRAE_loss * args.weight_drae

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss
            _, predicts = torch.max(outputs, 1)
            correct_num = torch.eq(predicts, targets).sum()
            correct_sum += correct_num

            running_acc = correct_sum.float() / float((batch_i + 1) * args.batch_size)
            pbar.set_postfix({'Loss': f'{train_loss/(batch_i+1):.3f}', 'Acc': f'{running_acc:.4f}'})

        train_acc = correct_sum.float() / float(train_dataset.__len__())
        train_loss = train_loss / iter_cnt
        elapsed = (time() - start_time) / 60

        print('[Epoch %d] Train time:%.2f, Training accuracy:%.4f. Loss: %.3f LR:%.6f' %
              (i, elapsed, train_acc, train_loss, optimizer.param_groups[0]["lr"]))

        scheduler.step()

        pre_labels = []
        gt_labels = []
        with torch.no_grad():
            val_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            model.eval()
            vbar = tqdm(val_loader, desc=f'Epoch {i}/{args.epochs} [Val]', ncols=100)
            for batch_i, (imgs, targets) in enumerate(vbar):
                with torch.amp.autocast('cuda'):
                    outputs, features = model(imgs.cuda())
                targets = targets.cuda()

                CE_loss = CE_criterion(outputs, targets)
                loss = CE_loss

                val_loss += loss
                iter_cnt += 1
                _, predicts = torch.max(outputs, 1)
                correct_or_not = torch.eq(predicts, targets)
                bingo_cnt += correct_or_not.sum().cpu()
                pre_labels += predicts.cpu().tolist()
                gt_labels += targets.cpu().tolist()

            val_loss = val_loss / iter_cnt
            val_acc = bingo_cnt.float() / float(val_dataset.__len__())
            val_acc = np.around(val_acc.numpy(), 4)
            f1 = f1_score(pre_labels, gt_labels, average='macro')
            total_socre = 0.67 * f1 + 0.33 * val_acc

            print("[Epoch %d] Validation accuracy:%.4f, Loss:%.3f, f1 %4f, score %4f" % (
            i, val_acc, val_loss, f1, total_socre))

            # 按验证精度保存最佳模型（无最低阈值限制）
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({'iter': i,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'best_acc': best_acc,
                            'best_score': best_score},
                           os.path.join('./checkpoint', "best_acc.pth"))
                print('Best acc model saved. best_acc: %.4f' % best_acc)
            # 按综合得分保存最佳模型
            if total_socre > best_score:
                best_score = total_socre
                torch.save({'iter': i,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'best_acc': best_acc,
                            'best_score': best_score},
                           os.path.join('./checkpoint', "best_score.pth"))
                print('Best score model saved. best_score: %.4f' % best_score)


if __name__ == "__main__":
    run_training()
