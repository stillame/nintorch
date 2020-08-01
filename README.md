# nintorch
A simple wrapper of Pytorch designed for researching and prototyping. <br>
Currently, work on progress. A lot of change and error will be expected. <br>

## To install:
```
pip install nintorch
```

## Directory tree:
```
nintorch____compress
	    |___dataset
	    |___ensemble
	    |___hyper
	    |___image
	    |___model
	    |___model_zoo
	    |___test
		|___type
```

## Requirements:
```
python == 3.7
torch == 1.5
torchvision == 0.6.0
apex == 0.1
ninstd == 0.4
pandas == 1.0.3
matplotlib == 3.1.3
loguru == 0.5.0
pytest == 5.4.3
opencv-python ==
numpy == 1.18.1
```

## Example:
A section from example/vgg16bn-cifar10.py
```python
device = torch_cpu_or_gpu()
train_loader, test_loader = load_dataset(
num_train_batch=args.train_batch,
num_test_batch=args.test_batch,
num_extra_batch=0, num_worker=8, 
dataset='cifar10', roof=DATASET_LOC,
transforms_list=crop_filp_normalize_transforms(
    CIFAR10_MEAN, CIFAR10_STD, 32, 4))

model = VGG16BN(in_chl=3).to(device)
model.init_weight_bias()
optim = optim.AdamW(
model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = MultiStepLR(optim, STEP_DOWN_EPOCHS, gamma=0.1)
loss_fuct = nn.CrossEntropyLoss()

if args.half:
trainer = HalfTrainer(
    model, optim=optim, loss_func=loss_fuct,
    train_loader=train_loader, test_loader=test_loader,
    scheduler=scheduler, writer=writer)
trainer.to_half(opt_level='O2')
else:
trainer = Trainer(
model, optim=optim, loss_func=loss_fuct,
train_loader=train_loader, test_loader=test_loader,
scheduler=scheduler, writer=writer)

trainer.warm_up_lr(args.lr, verbose=VERBOSE)
for i in range(1, args.epoch):
trainer.train_an_epoch(verbose=VERBOSE)
trainer.eval_an_epoch('test', verbose=VERBOSE)3c

trainer.dfs['train'].to_csv(save_train_csv_path)
trainer.dfs['test'].to_csv(save_test_csv_path)
```

## License:
```
MIT License with some exceptions.
Some functions or classes might be modified from other sources.
Please look at comments or docstrings within the functions and classes.
```

## TODO:
- swap optim on fly, early stop when best acc cannot improve the accuarcy.
- Automatically generate writer?
- fine-grained forwarding.
- if best then saving the model.
- adding input into the checking to cover more assert.
- look at other works kaggler.
- Adding optuna hyper-tuning.
- Checking is that cover the not image dataset.
- Adding grad-cam.
- Adding unitest.
- Adding type hinting.
- Adding example.
- Make trainer generalize to the more than one inputs and outputs.
- Update tqdm to display metric and loss.
- Update Trainer before epoch, after epoch, before batch, after batch, before .backward and before optim. Using concept from distiller.
