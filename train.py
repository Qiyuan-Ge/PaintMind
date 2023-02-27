import pathlib
import argparse
import paintmind as pm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epoch", type=int, default=20, help="training epochs")
    parser.add_argument("--bs", type=int, default=4, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="adamw: learning rate")
    parser.add_argument("--wd", type=float, default=1e-4, help="adamw: weight decay")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="warmup epochs")
    parser.add_argument("--lr_decay", type=float, default=0.9, help="apply lr decay or not")
    parser.add_argument("--data_path", type=str, default='none')
    parser.add_argument("--ema_decay", type=float, default=0.99, help="Exponential Moving Average, default: 0.99")
    arg = parser.parse_args()
    print(arg)
    
    if arg.data_path == 'none':
        root = pathlib.Path.cwd() / 'dataset' / 'coco'
        data_path = pathlib.Path(root)
        data_path.mkdir(exist_ok=True, parents=True)
    else:
        root = arg.data_path
        
    res_path = pathlib.Path.cwd() / "result"
    res_path.mkdir(exist_ok=True)
    print(f"result folder path: {res_path}")
    
    dataset = pm.datasets.coco(
        root=root,
        dataType='train2017', 
        annType='captions',
        transform=pm.create_transform(resize=320, crop_size=256, crop=True, is_train=True),
    )
        
    model = pm.create_model()
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'number of learnable parameters: {n_parameters//1e6}M')            
    
    dataloader = pm.TxtImgDataloader(dataset, batch_size=arg.bs, shuffle=True, num_workers=0, pin_memory=False)

    trainer = pm.PaintMindTrainer(
        model            =model, 
        lr               =arg.lr, 
        wd               =arg.wd, 
        dataloader       =dataloader,
        num_epochs       =arg.n_epoch,
        lr_min           =0.5*arg.lr,
        warmup_steps     =arg.wawarmup_steps,
        warmup_lr_init   =0.1*arg.lr,
        ema_decay        =arg.ema_decay,
        max_grad_norm    =1.0, 
        text_max_length  =77,
        checkpoint_path  =res_path,
        sample_interval  =1000,
        vqf4_config_path ='./logs/vq_f4/config.yaml',
        vqf4_pretrained_path=r'./logs/vq_f4/model.ckpt',
    )
    
    trainer.train()


if __name__ == "__main__":
    main()