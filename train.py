# Train the model

from model import UNet
from load import SegDataset, colormap2label
from mxnet import init, autograd, nd, gluon
from mxnet.gluon import loss as gloss, data as gdata, utils as gutils
import mxnet as mx
import sys
import os
import time
from settings import parse_opts, COLORMAP


def _get_batch(batch, ctx, is_even_split=True):
    features, labels = batch
    if labels.dtype != features.dtype:
        labels = labels.astype(features.dtype)
    return gutils.split_and_load(features, ctx, even_split=is_even_split), gutils.split_and_load(labels, ctx, even_split=is_even_split), features.shape[0]


def evaluate_accuracy(data_iter, net, ctx=[mx.cpu()]):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    acc_sum, n = nd.array([0]), 0
    for batch in data_iter:
        features, labels, _ = _get_batch(batch, ctx)
        for x, y in zip(features, labels):
            y = y.astype('float32')
            acc_sum += (net(x).argmax(axis=1) == y).sum().copyto(mx.cpu())
            n += y.size
        acc_sum.wait_to_read()
    return acc_sum.asscalar() / n


def train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs, log_dir='./', checkpoints_dir='./checkpoints'):
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    with open(log_dir + os.sep + 'UNet_log.txt', 'w') as f:
        print('training on', ctx, file=f)
        print('training on', ctx)
        if isinstance(ctx, mx.Context):
            ctx = [ctx]
        for epoch in range(num_epochs):
            train_l_sum, train_acc_sum, n, m, start = 0.0, 0.0, 0, 0, time.time()
            for i, batch in enumerate(train_iter):
                xs, ys, batch_size = _get_batch(batch, ctx)
                ls = []
                with autograd.record():
                    y_hats = [net(x) for x in xs]
                    ls = [loss(y_hat, y) for y_hat, y in zip(y_hats, ys)]
                for l in ls:
                    l.backward()
                trainer.step(batch_size)
                train_l_sum += sum([l.sum().asscalar() for l in ls])
                n += sum([l.size for l in ls])
                train_acc_sum += sum([(y_hat.argmax(axis=1) == y).sum().asscalar() for y_hat, y in zip(y_hats, ys)])
                m += sum([y.size for y in ys])
            test_acc = evaluate_accuracy(test_iter, net, ctx)
            print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.3f sec'
                  % (epoch + 1, train_l_sum / n, train_acc_sum / m, test_acc, time.time() - start), file=f)
            print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.3f sec'
                  % (epoch + 1, train_l_sum / n, train_acc_sum / m, test_acc, time.time() - start))

            if epoch != 1 and (epoch + 1) % 50 == 0:
                net.save_parameters(checkpoints_dir + os.sep + 'epoch_%04d_model.params' % (epoch + 1))


if __name__ == '__main__':
    args = parse_opts()
    print(args)

    if args.gpu_id is None:
        ctx = [mx.cpu()]
    else:
        ctx = [mx.gpu(i) for i in range(len(args.gpu_id))]
        s = ''
        for i in args.gpu_id:
            s += str(i) + ','
        s = s[:-1]
        os.environ['MXNET_CUDA_VISIBLE_DEVICES'] = s
        os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

    root_dir = os.path.join(args.data_dir)
    net = UNet(64, args.num_classes)
    net.initialize(init=init.Xavier(), ctx=ctx)
    print(net)

    batch_size = args.batch_size

    num_workers = 0 if sys.platform.startswith('win') else 4
    train_imgs = SegDataset(is_train=True,
                            is_crop=args.is_crop,
                            crop_size=(args.crop_height, args.crop_width),
                            root=root_dir,
                            colormap2label=colormap2label)
    test_imgs = SegDataset(is_train=False,
                           is_crop=args.is_crop,
                           crop_size=(args.crop_height, args.crop_width),
                           root=root_dir,
                           colormap2label=colormap2label)
    train_iter = gdata.DataLoader(train_imgs,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  last_batch='keep')
    test_iter = gdata.DataLoader(test_imgs,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=num_workers,
                                 last_batch='keep')

    loss = gloss.SoftmaxCrossEntropyLoss(axis=1)
    if args.optimizer == 'sgd':
        optimizer_params = {'learning_rate': args.learning_rate, 'momentum': args.momentum}
    else:
        optimizer_params = {'learning_rate': args.learning_rate}
    trainer = gluon.Trainer(net.collect_params(),
                            args.optimizer,
                            optimizer_params)
    train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs=args.num_epochs, log_dir=args.log_dir)
