
import sys 
sys.path.append("/home/khatpn/KT/take_home_test")

from models.mobilenetv2 import build_model, build_lr_scheduler, build_optimizer
from data.dataloader import MyDataModule
from utils import ClsMetric, MultilabelMetric, AverageMeter, MultilabelPostprocessing, pretty_output, convert_cuda, save_model

import os
import argparse
import yaml
import torch
import copy
import time

from tqdm import tqdm
from torch import nn
import shutil

def validate(net: nn.Module, 
             dataloader: torch.utils.data.DataLoader,
             metric_monitor: ClsMetric,
             criteria: nn.modules.loss._Loss,
             post_process: object,
             is_test=False,
             epoch=0):
    # if arch_config is not None: Do not random samples

    # switch to train mode
    net.eval()

    data_time = AverageMeter()
    losses =  AverageMeter()
    metric_monitor.reset()

    nBatch = len(dataloader)
    _str = "Evaluate" if not is_test else "Test"
    with tqdm(
        total=nBatch,
        desc="{} Epoch #{}".format(_str, epoch + 1),
    ) as t:
        end = time.time()
        for i, batches in enumerate(dataloader):
            data_time.update(time.time() - end)

            inputs, labels, labels_one_hot, _  = convert_cuda(batches)
            with torch.no_grad():
                outputs = net(inputs)
            loss = criteria(outputs, labels) if not post_process else criteria(outputs, labels_one_hot) 

            if isinstance(loss, dict):
                loss = loss['loss']

            if post_process:
                outputs = post_process(outputs)

            metric_monitor(outputs, labels) if not post_process else metric_monitor(outputs, labels_one_hot)
            losses.update(loss, batches[0].size(0))

            t.set_postfix(
                {
                    "loss": losses.avg.item(),
                    "data_time": data_time.avg,
                }
            )
            t.update(1)
            end = time.time()
            
    

    return losses.avg.item(), metric_monitor.get_metric(reduction=False)


def train_one_epoch(net: nn.Module, 
                    train_loader: torch.utils.data.DataLoader,
                    metric_monitor: ClsMetric,
                    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
                    optimizer: torch.optim.Optimizer,
                    criteria: nn.modules.loss._Loss,
                    post_process: object,
                    epoch):
    # if arch_config is not None: Do not random samples

    # switch to train mode
    net.train()

    data_time = AverageMeter()
    losses =  AverageMeter()
    metric_monitor.reset()

    nBatch = len(train_loader)
    with tqdm(
        total=nBatch,
        desc="Train Epoch #{}".format(epoch + 1),
    ) as t:
        end = time.time()
        for i, batches in enumerate(train_loader):
            data_time.update(time.time() - end)
            lr_scheduler.step()
            optimizer.zero_grad()

            inputs, labels, labels_one_hot, _  = convert_cuda(batches)
            outputs = net(inputs)
            loss = criteria(outputs, labels) if not post_process else criteria(outputs, labels_one_hot) 



            if isinstance(loss, dict):
                loss = loss['loss']
            loss.backward()
            optimizer.step()

            if post_process:
                outputs = post_process(outputs)

            metric_monitor(outputs, labels) if not post_process else metric_monitor(outputs, labels_one_hot)
            losses.update(loss, batches[0].size(0))

            t.set_postfix(
                {
                    "loss": losses.avg.item(),
                    **metric_monitor.get_metric(),
                    "lr": optimizer.param_groups[0]['lr'],
                    "data_time": data_time.avg,
                }
            )
            t.update(1)
            end = time.time()
            
    

    return losses.avg.item(), metric_monitor.get_metric()

def train(args, config, datamodule, model, optimizer, lr_scheduler, criterion, label_list):
    pass
    max_epoch = config['Global']['max_epoch']
    eval_step = config['Global']['eval_step']
    metric_monitor = ClsMetric(main_indicator="f1", n_classes=len(label_list)) if not args.is_multilabel else \
                    MultilabelMetric(main_indicator="f1", n_classes=len(label_list))

    post_processing = MultilabelPostprocessing(0.3) if args.is_multilabel else None
    best_acc = -1

    for epoch in range(max_epoch):

        loss, metrics = train_one_epoch(model,
                                        datamodule.train,
                                        metric_monitor,
                                        lr_scheduler,
                                        optimizer,
                                        criterion,
                                        post_processing,
                                        epoch=epoch
                                        )
        
        if (epoch + 1) % eval_step == 0:
            val_loss, val_metrics_no_reduce = validate(model, datamodule.val, metric_monitor, criterion, post_processing, is_test=False, epoch=epoch)
            pretty_output(label_list, val_metrics_no_reduce)
            val_metrics = metric_monitor.reduce(val_metrics_no_reduce) 
            
            #Save best epoch
            val_acc = val_metrics["f1"]
            is_best = val_acc >= best_acc
            best_acc = max(best_acc, val_acc) 

            model_name = "epoch_{:03d}".format(epoch)
            model_path = save_model({
                    "epoch": epoch,
                    "best_acc": best_acc,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": lr_scheduler.state_dict(),
                    "state_dict": model.state_dict() ,
                    },
                    save_path=f"{args.log_dir}/checkpoint/",
                    is_best=is_best,
                    model_name=model_name
                )


def test(args, config, datamodule, model, criterion, label_list, is_test=True):
    metric_monitor = ClsMetric(main_indicator="f1", n_classes=len(label_list)) if not args.is_multilabel else \
                    MultilabelMetric(main_indicator="f1", n_classes=len(label_list))

    post_processing = MultilabelPostprocessing(0.3) if args.is_multilabel else None
    test_loss, test_metrics_no_reduce = validate(model, datamodule.test if is_test else datamodule.val, metric_monitor, criterion, post_process=post_processing, is_test=is_test)
    pretty_output(label_list, test_metrics_no_reduce)

    return test_metrics_no_reduce

def main(args, config):

    #Dataset
    label_path = config['Global']['label_list']
    label_list = [x.strip() for x in open(label_path)]
    classes_name = {x:i for i, x in enumerate(label_list)}
    config['Global']['n_classes'] = len(label_list)
    config['Model']['n_classes'] = len(label_list)
    is_multilabel = config['Model']['multilabel']
    args.is_multilabel = is_multilabel

    data_config = copy.deepcopy(config['Datasets'])
    data_config['classes_name'] = classes_name
    datamodule = MyDataModule(data_config)
    train_dataloader = datamodule.train
    nBatch = len(train_dataloader)

    #Model, opt, lr, metric, engine
    model = build_model(**config["Model"]) 
    model.cuda()
    opt_config = copy.deepcopy(config['Optimizer'])
    optimizer = build_optimizer(model, opt_config)
    lr_scheduler = build_lr_scheduler(optimizer, config, nBatch)

    criterion = nn.CrossEntropyLoss() if not is_multilabel else nn.BCELoss()


    if not args.run_test:
        train(args, config, datamodule, model, optimizer, lr_scheduler, criterion, label_list)

    test(args, config, datamodule, model, criterion, label_list)


def get_last_version(path):
    import glob 
    files = glob.glob(path + "/version_*")
    if not files:
        return -1
    version = [int(x.split('/')[-1].split('_')[-1]) for x in files]

    return max(version)

def merge_config(args, config):
    config['Model']['checkpoint'] = args.ckpt

    return config

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Train classification")
    parser.add_argument("config", type=str, default=None, help="path to config")
    parser.add_argument("--run_test", action="store_true", help="run train or test")
    parser.add_argument("--log_dir", type=str, default="log_train", help="path to dir")
    parser.add_argument("--ckpt", type=str, default=None, help="path to dir")

    args = parser.parse_args()
    
    if not args.run_test:
        version = get_last_version(args.log_dir) + 1 
        args.log_dir = f'{args.log_dir}/version_{version}'
        os.makedirs(args.log_dir, exist_ok=True)
        shutil.copy(args.config, args.log_dir + "/config.yaml")


    else:
        args.config = os.path.join(args.log_dir, "config.yaml")
    

    with open(args.config, "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader) 
    config = merge_config(args, config)

    print(config["Model"]["multilabel"])

    main(args, config)

