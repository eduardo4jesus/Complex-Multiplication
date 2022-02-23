.RECIPEPREFIX = >
.DEFAULT_GOAL := extensions

extensions:
> python pytorch_extensions/setup.py install

clean:
> rm -rf *.egg-info build dist .ipynb_checkpoints

.PHONY: extension clean