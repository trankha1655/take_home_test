import torchvision.models as models
import torch 

from torchvision.models import MobileNet_V2_Weights
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


def build_model(n_classes, pretrain=False, checkpoint=None, multilabel=False, *args, **kwargs):
    weights = MobileNet_V2_Weights.IMAGENET1K_V1
    model = models.mobilenet_v2(weights=weights if pretrain else None)

    if multilabel:
        model.classifier[1] = nn.Sequential(
                nn.Linear(model.last_channel, n_classes),
                nn.Sigmoid()
        )

    else:
        model.classifier[1] = nn.Linear(model.last_channel, n_classes)


    if checkpoint is not None:
        ckpt_path = checkpoint
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        print("Loaded model from", ckpt_path)

    return model

def build_optimizer(net: nn.Module, config: dict):

    net_params = list(net.parameters())
    config.update({"net_params": net_params})

    def _builder(net_params, opt_param, init_lr, weight_decay, **kwargs):

        net_params = [
                {"params": net_params, "weight_decay": weight_decay},
            ]
        
        beta1, beta2 = opt_param.get("beta1", 0.9), opt_param.get("beta2", 0.999)
        optimizer = torch.optim.Adam(net_params, init_lr, (beta1, beta2))

        return optimizer 
    
    return _builder(**config)

def build_lr_scheduler(optimizer, config, N=10):    #N is number of data batches
    
    eta_min = config['LR_scheduler']['eta_min']
    n_warmup_steps = config['LR_scheduler']['warm_up_epoch'] * N
    T_max = config['Global']['max_epoch'] * N

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=n_warmup_steps
    )

    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=T_max-n_warmup_steps, eta_min=eta_min)

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[n_warmup_steps]
    )

    return scheduler
    
    



    
if __name__ == "__main__":
    pass 


    model = build_model(10, pretrain=False)
    model.train()
    

    # for param in model.parameters():
    #     print(param.shape)


    opt_config = {
        "net_params": list(model.parameters()),
        "opt_param": {'beta1': 0.9, 'beta2': 0.999},
        "init_lr": 0.001,
        "weight_decay": 0.00003,

    }

    opt = build_optimizer(**opt_config)
    lr_scheduler =  build_lr_scheduler(opt, n_warmup_steps=100, T_max=20000)

    for i in range(20000):
        lr_scheduler.step()
        lr = opt.param_groups[0]['lr']
        if i >19500 or i <200:
            print(i, lr) 