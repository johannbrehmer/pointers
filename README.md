# Pointers: tools, techniques and tips that make me feel more productive

Johann Brehmer 2020
### Project organization

- Claudio Jolowicz has lots of good advice on how to set up a Python project at his [Guide to hypermodern Python](https://cjolowicz.github.io/posts/hypermodern-python-01-setup/).
- My typical repository setup is demonstrated in [example_repo](./example_repo/). It's not at the same level as the hypermodern Python guide, but I felt comfortable with that setup.

### Coding

- PyCharm and VS Code are both great IDEs. 
- As far as Python style goes, I try to follow PEP8 (of course) and the [black code style](https://black.readthedocs.io/) with a slightly increased line length.
- My most visited website of 2019 and 2020 is probably [this StackOverflow question](https://stackoverflow.com/questions/53004311/how-to-add-conda-environment-to-jupyter-lab) on how to add conda environments to Jupyter Lab.

### Machine learning

- For experiment management, I recently used [sacred](https://github.com/IDSIA/sacred). See [example_repo/experiments/experiment.py](./example_repo/experiments/experiment.py). Alternatives I want to try include [MLFlow](https://mlflow.org/), which seems to be more actively developed. 
- Some advice on optimizing PyTorch code at [this tweet](https://twitter.com/karpathy/status/1299921324333170689).

### Running on HPC

- I'm not very knowledgable on this, but my workflow involved synching code between my laptop and the cluster through git and data with rsync.
- I like to submit jobs arrays (`sbatch --array 0-99 run.sh`, see [example_repo/experiments/hpc/run.sh]).
### Visualization

(I have too strong opinions on this, but here you go...)

- Colors, line styles etc should be chosen with semantics, accessability, and consistency in mind. Lots of good points on choosing discrete color palettes can be found in [Lisa Charlotte Rost's guide](https://blog.datawrapper.de/beautifulcolors/). Color maps should be perceptually uniform ones ([this matplotlib doc](https://matplotlib.org/3.3.3/tutorials/colors/colormaps.html) isn't bad as a starting point). To pick colors (and check for greyscale representations and accessibility issues), I like [coolors.co](https://coolors.co).
- Figures should have consistent font sizes, roughly matching the paper text size. Authors that use small figures with tiny tiny captions are terrible people and eat baby kittens.
- Figures should also have consistent margins. Don't choose margins automatically through `plt.tight_layout()`, but set them consciously. I am partial to square panels inside square canvasses. I've been using some (rather ugly) helper functions to achieve that, you can find them in [plot_layout.py](./plot_layout.py).

### Writing

- VS Code is great for LaTeX as well.
- The best citation manager I found so far is [BibDesk](https://bibdesk.sourceforge.io/). Unfortunately, it's MacOS only. Mendeley frustrated me because of its poor BibTeX export (so many issues with special characters), JabRef with its performance issues on Mac.


### Acknowledgements

Thanks to Alexander Held and Kyle Cranmer for teaching me many of these things.
