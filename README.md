# nintorch
A simple wrapper of Pytorch designed for researching and prototyping. <br>
Currently, work on progress. A lot of change and error will be expected. <br>

## To install:
```
pip install nintorch
```

## Example:
```
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
