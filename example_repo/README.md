# Example repository

### Structure

- [example/](./example/) contains the main functionality: models, training procedures, and so on. In this example repo that's just some boilerplate flow code and a trainer class.
- [experiments/](./experiments/) has code for experiments, operated with `sacred`. This includes dataset-specific code, run data, and HPC scripts.
- [utils/](./utils/) includes helper scripts to convert the code to black code style and to sync results between local repository and HPC.
- This root folder includes the conda environment specification, the .gitignore, the license file and this readme.

### Acknowledgements

The normalizing flow code, some of the datasets, and some utility functions are blatantly stolen from the excellent [nflows library](https://github.com/bayesiains/nflows) and the [neural spline flows codebase](https://github.com/bayesiains/nsf) by Conor Durkan, Artur Bekasov, Iain Murray, and George Papamakarios.
 