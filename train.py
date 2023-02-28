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
    
    dataset = pm.load_dataset(
        name='coco',
        root=root, 
        transform=pm.create_transform(resize=320, crop_size=256, crop=True, is_train=True),
    )
        
    model = pm.create_model()
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'number of learnable parameters: {n_parameters//1e6}M')            
    
    dataloader = pm.TxtImgDataloader(dataset, batch_size=arg.bs, shuffle=True, num_workers=0, pin_memory=False)

    trainer = pm.PaintMindTrainer(
        model                       = model,
        lr                          = arg.lr, 
        wd                          = arg.wd, 
        dataloader                  = dataloader,
        num_epochs                  = arg.n_epoch,
        lr_min                      = 0.5*arg.lr,
        warmup_steps                = arg.warmup_steps,
        warmup_lr_init              = 0.1*arg.lr,
        ema_decay                   = None,
        max_grad_norm               = 1.0, 
        text_max_length             = 256,
        checkpoint_path             = res_path,
        sample_interval             = 1000,
        save_every_n_step           = 1000,
        first_stage_config_path     = './models/first_stage_models/kl-f4/config.yaml',
        first_stage_pretrained_path = './models/first_stage_models/kl-f4/model.ckpt',
        gradient_accumulation_steps = 8,
    )
    
    trainer.train()


if __name__ == "__main__":
    main()