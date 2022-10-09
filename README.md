# Pointers

*Johann Brehmer (mail@johannbrehmer.de), 2020*

This is a list of tools, techniques, and tips that made me feel more productive. In all of these areas I have much to learn, if you think "X is stupid, you should do Y instead", please let me know.

### Project organization

- Claudio Jolowicz wrote a [great guide on how to set up a Python project](https://cjolowicz.github.io/posts/hypermodern-python-01-setup/).
- My own repositories are far from these best practices, but worked well for me for testing out ideas fast. I included an example of my repository structure in [example_repo](./example_repo/).

### Coding

- [Black](https://black.readthedocs.io/) (but with `-l 120`, unless you're Gutenberg) is neat.
- My most visited website of 2019 and 2020 is probably [this StackOverflow question](https://stackoverflow.com/questions/53004311/how-to-add-conda-environment-to-jupyter-lab) on how to add conda environments to Jupyter Lab.

### Machine learning

- Some advice on optimizing PyTorch code at [this tweet](https://twitter.com/karpathy/status/1299921324333170689).
- For config management, I have used [sacred](https://github.com/IDSIA/sacred) and [hydra](https://github.com/facebookresearch/hydra), preferring the latter.
- [MLFlow](https://mlflow.org/) is my favourite tool for experiment tracking so far.
- For hyperparameter tuning, I like [Optuna](https://optuna.readthedocs.io/en/stable/).

### Visualization

(I have too strong opinions on this, but here you go...)

- Colors, line styles etc should be chosen with semantics, accessability, and consistency in mind. Lots of good points on choosing discrete color palettes can be found in [Lisa Charlotte Rost's guide](https://blog.datawrapper.de/beautifulcolors/). Color maps should be perceptually uniform, [this matplotlib doc](https://matplotlib.org/3.3.3/tutorials/colors/colormaps.html) is a good starting point.
- To pick colors (and check for greyscale representations and accessibility issues), I like [coolors.co](https://coolors.co). [Palettable](https://jiffyclub.github.io/palettable/) is fun.
- Figures should have consistent font sizes, roughly matching the paper text size. Authors that use small figures with tiny tiny captions are terrible people and eat kittens.
- Figures should also have consistent margins. Don't choose margins automatically through `plt.tight_layout()`, but set them consciously. I am partial to square panels inside square canvasses. I've been using some (rather ugly) helper functions to achieve that, you can find them in [setup_figures.py](./example_repo/experiments/evaluation/setup_figures.py), I'm sure there is a prettier way though.

### Writing

- The best citation manager I found so far are [BibDesk](https://bibdesk.sourceforge.io/) (local, free, only MacOS) and [Paperpile](https://paperpile.com/) (cloud-based, good for sharing folders). Mendeley frustrated me because of its poor BibTeX export, JabRef with its performance issues on Mac.

### Acknowledgements

Thanks to Alexander Held, Kyle Cranmer, and Siddharth Mishra-Sharma for teaching me many of these things.
