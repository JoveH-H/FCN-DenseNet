import gc
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from dataset import MyData
from model import Deconv
import densenet
import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', default='./resources/images/data/train/')  # training dataset
parser.add_argument('--val_dir', default='./resources/images/data/test/')  # test dataset
parser.add_argument('--check_dir', default='./parameters')  # save checkpoint parameters
parser.add_argument('--q', default='densenet121')  # save checkpoint parameters
parser.add_argument('--b', type=int, default=4)  # batch size
parser.add_argument('--e', type=int, default=100)  # epoches
parser.add_argument('--svae_interval', type=int, default=5)  # svae interval
opt = parser.parse_args()


def validation(feature, net, loader):
    feature.eval()
    net.eval()
    total_loss = 0
    for ib, (data, lbl) in enumerate(loader):
        inputs = Variable(data).cuda()
        lbl = Variable(lbl.float().unsqueeze(1)).cuda()

        feats = feature(inputs)
        msk = net(feats)

        loss = F.binary_cross_entropy_with_logits(msk, lbl)
        total_loss += loss.item()
    feature.train()
    net.train()
    return total_loss / len(loader)


def main():
    train_dir = opt.train_dir
    val_dir = opt.val_dir
    check_dir = opt.check_dir + '_' + opt.q
    bsize = opt.b
    epoch_sum = opt.e
    svae_interval = opt.svae_interval

    if not os.path.exists(check_dir):
        os.mkdir(check_dir)

    feature = getattr(densenet, opt.q)(pretrained=True)
    feature.cuda()
    deconv = Deconv(opt.q)
    deconv.cuda()

    train_loader = torch.utils.data.DataLoader(MyData(train_dir, transform=True, crop=False, hflip=False, vflip=False),
                                               batch_size=bsize, shuffle=True, num_workers=1, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(MyData(val_dir,  transform=True, crop=False, hflip=False, vflip=False),
                                             batch_size=bsize, shuffle=False, num_workers=1, pin_memory=True)

    optimizer = torch.optim.AdamW([
        {'params': feature.parameters(), 'lr': 1e-3},
        {'params': deconv.parameters(), 'lr': 1e-3},
    ])

    min_loss = 10000.0
    for it in range(epoch_sum):
        step_len = len(train_loader)
        for ib, (data, lbl) in enumerate(train_loader):
            inputs = Variable(data).cuda()
            lbl = Variable(lbl.float().unsqueeze(1)).cuda()
            feats = feature(inputs)
            msk = deconv(feats)
            loss = F.binary_cross_entropy_with_logits(msk, lbl)

            deconv.zero_grad()
            feature.zero_grad()

            loss.backward()

            optimizer.step()

            # 显示进度
            step_now = ib + 1
            step_schedule_num = int(40 * step_now / step_len)
            epoch_now = it + 1
            print("\r", end="")
            print("epoch: {}/{} step: {}/{} [{}{}] - loss: {:.5f}".format(epoch_now, epoch_sum,
                                                                          step_now, step_len,
                                                                          ">" * step_schedule_num,
                                                                          "-" * (40 - step_schedule_num),
                                                                          loss.item()), end="")
            sys.stdout.flush()

            # 清除变量和内存
            del inputs, msk, lbl, loss, feats
            gc.collect()

        print("\r")

        if epoch_now % svae_interval == 0:
            val_loss = validation(feature, deconv, val_loader)
            if val_loss < min_loss:
                filename = ('{}/deconv_model.pth'.format(check_dir))
                torch.save(deconv.state_dict(), filename)
                filename = ('{}/feature_model.pth'.format(check_dir))
                torch.save(feature.state_dict(), filename)
                print('epoch: {} val loss: {:.5f} save model'.format(epoch_now, val_loss))
                min_loss = val_loss
            else:
                print('epoch: {} val loss: {:.5f} pass'.format(epoch_now, val_loss))


if __name__ == "__main__":
    main()
