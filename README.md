# Pointers

*Johann Brehmer (mail@johannbrehmer.de), 2020*

This is a list of tools, techniques, and tips that made me feel more productive. In all of these areas I have much to learn, if you think "X is stupid, you should do Y instead", please let me know.

### Project organization

- Claudio Jolowicz wrote a great guide on how to set up a Python project: [Guide to hypermodern Python](https://cjolowicz.github.io/posts/hypermodern-python-01-setup/).
- My own repositories are far from these best practices. Nevertheless, they worked well for me for testing out ideas fast. I included an example of my repository structure in [example_repo](./example_repo/).

### Coding

- PyCharm and VS Code are both good IDEs.
- As far as Python style goes, I try to follow PEP8 (of course) and the [black code style](https://black.readthedocs.io/), though I like to increase the line length to 120 characters.
- My most visited website of 2019 and 2020 is probably [this StackOverflow question](https://stackoverflow.com/questions/53004311/how-to-add-conda-environment-to-jupyter-lab) on how to add conda environments to Jupyter Lab.

### Machine learning

- Some advice on optimizing PyTorch code at [this tweet](https://twitter.com/karpathy/status/1299921324333170689).
- For experiment management, I have lately been using [sacred](https://github.com/IDSIA/sacred). See [experiment.py](./example_repo/experiments/experiment.py). Alternatives I want to try include [MLFlow](https://mlflow.org/), which seems to be more actively developed. 
- [Optuna](https://optuna.readthedocs.io/en/stable/) is my personal favorite hyperparameter scan library.

### HPC

- I'm not very knowledgable on this, but my workflow involved synching code between my laptop and the cluster through git and data with rsync.
- I like to submit jobs arrays a la `sbatch --array 0-99 run.sh`, see [run.sh](example_repo/experiments/hpc/run.sh).

### Visualization

(I have too strong opinions on this, but here you go...)

- Colors, line styles etc should be chosen with semantics, accessability, and consistency in mind. Lots of good points on choosing discrete color palettes can be found in [Lisa Charlotte Rost's guide](https://blog.datawrapper.de/beautifulcolors/). Color maps should be perceptually uniform, [this matplotlib doc](https://matplotlib.org/3.3.3/tutorials/colors/colormaps.html) is a good starting point.
- To pick colors (and check for greyscale representations and accessibility issues), I like [coolors.co](https://coolors.co). [Palettable](https://jiffyclub.github.io/palettable/) is fun.
- Figures should have consistent font sizes, roughly matching the paper text size. Authors that use small figures with tiny tiny captions are terrible people and eat kittens.
- Figures should also have consistent margins. Don't choose margins automatically through `plt.tight_layout()`, but set them consciously. I am partial to square panels inside square canvasses. I've been using some (rather ugly) helper functions to achieve that, you can find them in [setup_figures.py](./example_repo/experiments/evaluation/setup_figures.py), I'm sure there is a prettier way though.

### Writing and LaTeX

- VS Code is great for LaTeX as well.
- The best citation manager I found so far is [BibDesk](https://bibdesk.sourceforge.io/). Unfortunately, it's only available for MacOS. Mendeley frustrated me because of its poor BibTeX export (so many issues with special characters), JabRef with its performance issues on Mac.

### Acknowledgements

Thanks to Alexander Held, Kyle Cranmer, and Siddharth Mishra-Sharma for teaching me many of these things.
