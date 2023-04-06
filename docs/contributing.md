# Contributing to liionpack

If you'd like to contribute to liionpack (thanks!), please have a look at the [guidelines below](#workflow).

If you're already familiar with our workflow, maybe have a quick look at the [pre-commit checks](#pre-commit-checks) directly below.

## Pre-commit checks

Fork the repository and create a pull request. Github actions should check that tests are passing.

### Installing and using pre-commit

`liionpack` uses a set of `pre-commit` hooks and the `pre-commit` bot to format and prettify the codebase. The hooks can be installed locally using -

```bash
pip install pre-commit
pre-commit install
```

This would run the checks every time a commit is created locally. The checks will only run on the files modified by that commit, but the checks can be triggered for all the files using -

```bash
pre-commit run --all-files
```

If you would like to skip the failing checks and push the code for further discussion, use the `--no-verify` option with `git commit`.

## Workflow

We use [GIT](https://en.wikipedia.org/wiki/Git) and [GitHub](https://en.wikipedia.org/wiki/GitHub) to coordinate our work. When making any kind of update, we try to follow the procedure below.

### A. Before you begin

1. Create an [issue](https://guides.github.com/features/issues/) where new proposals can be discussed before any coding is done.
2. Create a [branch](https://help.github.com/articles/creating-and-deleting-branches-within-your-repository/) of this repo (ideally on your own [fork](https://help.github.com/articles/fork-a-repo/)), where all changes will be made
3. Download the source code onto your local system, by [cloning](https://help.github.com/articles/cloning-a-repository/) the repository (or your fork of the repository).
4. [Install](https://pybamm.readthedocs.io/en/latest/install/install-from-source.html) PyBaMM with the developer options.
5. [Test](#testing) if your installation worked, using the test script: `$ python -m unittest`.

You now have everything you need to start making changes!

### B. Writing your code

5. liionpack is developed in [Python](https://en.wikipedia.org/wiki/Python_(programming_language)), and makes heavy use of [NumPy](https://en.wikipedia.org/wiki/NumPy) (see also [NumPy for MatLab users](https://numpy.org/doc/stable/user/numpy-for-matlab-users.html) and [Python for R users](http://blog.hackerearth.com/how-can-r-users-learn-python-for-data-science)).
6. Make sure to follow our [coding style guidelines](#coding-style-guidelines).
7. Commit your changes to your branch with [useful, descriptive commit messages](https://chris.beams.io/posts/git-commit/): Remember these are publicly visible and should still make sense a few months ahead in time. While developing, you can keep using the GitHub issue you're working on as a place for discussion. [Refer to your commits](https://stackoverflow.com/questions/8910271/how-can-i-reference-a-commit-in-an-issue-comment-on-github) when discussing specific lines of code.
8. If you want to add a dependency on another library, or re-use code you found somewhere else, have a look at [these guidelines](#dependencies-and-reusing-code).

### C. Merging your changes with liionpack

9. [Test your code!](#testing)
10. liionpack has online documentation at http://liionpack.readthedocs.io/. To make sure any new methods or classes you added show up there, please read the [documentation](#documentation) section.
11. If you added a major new feature, perhaps it should be showcased in an [example notebook](#example-notebooks).
12. When you feel your code is finished, or at least warrants serious discussion, run the [pre-commit checks](#pre-commit-checks) and then create a [pull request](https://help.github.com/articles/about-pull-requests/) (PR) on [liionpack's GitHub page](https://github.com/pybamm-team/liionpack).
13. Once a PR has been created, it will be reviewed by any member of the community. Changes might be suggested which you can make by simply adding new commits to the branch. When everything's finished, someone with the right GitHub permissions will merge your changes into liionpack main repository.
14. The default branch is the `develop` branch and this is linked to the PyBaMM `develop` branch. We endeavour to make concurrent releases so that no breaking changes are introduced between the packages and at this point `develop` is merged into the `main` branch and this is pushed to PyPi.



## Coding style guidelines

liionpack follows the [PEP8 recommendations](https://www.python.org/dev/peps/pep-0008/) for coding style. These are very common guidelines, and community tools have been developed to check how well projects implement them.

### Ruff

We use [ruff](https://github.com/charliermarsh/ruff) to check our PEP8 adherence. To try this on your system, navigate to the liionpack directory in a console and type

```bash
python -m pip install pre-commit
pre-commit run ruff
```

ruff is configured inside the file `pre-commit-config.yaml`, allowing us to ignore some errors. If you think this should be added or removed, please submit an [issue](#issues)

When you commit your changes they will be checked against ruff automatically (see [infrastructure](#infrastructure)).

### Black

We use [black](https://black.readthedocs.io/en/stable/) to automatically configure our code to adhere to PEP8. Black can be used in two ways:

1. Command line: navigate to the liionpack directory in a console and type

```bash
black {source_file_or_directory}
```

2. Editor: black can be [configured](https://test-black.readthedocs.io/en/latest/editor_integration.html) to automatically reformat a python script each time the script is saved in an editor.

If you want to use black in your editor, you may need to change the max line length in your editor settings.

Even when code has been formatted by black, you should still make sure that it adheres to the PEP8 standard set by [Ruff](#ruff).

### Naming

Naming is hard. In general, we aim for descriptive class, method, and argument names. Avoid abbreviations when possible without making names overly long, so `mean` is better than `mu`, but a class name like `MyClass` is fine.

Class names are CamelCase, and start with an upper case letter, for example `MyOtherClass`. Method and variable names are lower case, and use underscores for word separation, for example `x` or `iteration_count`.


## Dependencies and reusing code

While it's a bad idea for developers to "reinvent the wheel", it's important for users to get a _reasonably sized download and an easy install_. In addition, external libraries can sometimes cease to be supported, and when they contain bugs it might take a while before fixes become available as automatic downloads to liionpack users.
For these reasons, all dependencies in liionpack should be thought about carefully, and discussed on GitHub.

Direct inclusion of code from other packages is possible, as long as their license permits it and is compatible with ours, but again should be considered carefully and discussed in the group. Snippets from blogs and [stackoverflow](https://stackoverflow.com/) can often be included without attribution, but if they solve a particularly nasty problem (or are very hard to read) it's often a good idea to attribute (and document) them, by making a comment with a link in the source code.


## Testing

All code requires testing. We use the [unittest](https://docs.python.org/3.3/library/unittest.html) package for our tests. (These tests typically just check that the code runs without error, and so, are more _debugging_ than _testing_ in a strict sense. Nevertheless, they are very useful to have!)

```bash
python -m unittest
```

### Writing tests

Every new feature should have its own test. To create ones, have a look at the `test` directory and see if there's a test for a similar method. Copy-pasting this is a good way to start.

Next, add some simple (and speedy!) tests of your main features. If these run without exceptions that's a good start! Next, check the output of your methods using any of these [assert methods](https://docs.python.org/3.3/library/unittest.html#assert-methods).


### Profiling

Sometimes, a bit of code will take much longer than you expect to run. In this case, you can set
```python
from IPython import embed; embed(); import ipdb; ipdb.set_trace()
```
as above, and then use some of the profiling tools. In order of increasing detail:
1. Simple timer. In ipython, the command
```
%time command_to_time()
```
tells you how long the line `command_to_time()` takes. You can use `%timeit` instead to run the command several times and obtain more accurate timings.
2. Simple profiler. Using `%prun` instead of `%time` will give a brief profiling report
3. Detailed profiler. You can install the detailed profiler `snakeviz` through pip:
```bash
pip install snakeviz
```
and then, in ipython, run
```
%load_ext snakeviz
%snakeviz command_to_time()
```
This will open a window in your browser with detailed profiling information.

## Documentation

liionpack is documented in several ways.

First and foremost, every method and every class should have a [docstring](https://www.python.org/dev/peps/pep-0257/) that describes in plain terms what it does, and what the expected input and output is.

These docstrings can be fairly simple, but can also make use of [reStructuredText](http://docutils.sourceforge.net/docs/user/rst/quickref.html), a markup language designed specifically for writing [technical documentation](https://en.wikipedia.org/wiki/ReStructuredText). For example, you can link to other classes and methods by writing ```:meth:`run()` ```.

In addition, we write a (very) small bit of documentation in separate reStructuredText files in the `docs` directory. Most of what these files do is simply import docstrings from the source code. But they also do things like add tables and indexes. If you've added a new class to a module, search the `docs` directory for that module's `.rst` file and add your class (in alphabetical order) to its index. If you've added a whole new module, copy-paste another module's file and add a link to your new file in the appropriate `index.rst` file.

Using [MKDocs](https://www.mkdocs.org/) the documentation in `docs` can be converted to HTML, PDF, and other formats. In particular, we use it to generate the documentation on http://liionpack.readthedocs.io/

### Building the documentation

MkDocs comes with a built-in dev-server that lets you preview your documentation as you work on it. Make sure you're in the same directory as the mkdocs.yml configuration file, and then start the server by running the following command:

```
mkdocs serve
```
And then visit the webpage served at http://127.0.0.1:8000. Each time a change to the documentation source is detected, the HTML is rebuilt and the browser automatically reloaded.

### Example notebooks

Major liionpack features are showcased in [Jupyter notebooks](https://jupyter.org/) stored in the [examples directory](examples/notebooks). Which features are "major" is of course wholly subjective, so please discuss on GitHub first!


## Citations

Our package is built on PyBaMM and we recommend that you use the citations functionality to give proper acknowledgment to contributing work.

We aim to recognize all contributions by automatically generating citations to the relevant papers on which different parts of the code are built.
These will change depending on what models and solvers you use.
Adding the command

```python3
pybamm.print_citations()
```

to the end of a script will print all citations that were used by that script. This will print bibtex information to the terminal; passing a filename to `print_citations` will print the bibtex information to the specified file instead.

When you contribute code to PyBaMM, you can add your own papers that you would like to be cited if that code is used. First, add the bibtex for your paper to citations.txt. Then, add the line

```python3
pybamm.citations.register("your_paper_bibtex_identifier")
```

wherever code is called that uses that citation (for example, in functions or in the `__init__` method of a class such as a model or solver).

## Benchmarks

A benchmark suite is located in the `benchmarks` directory at the root of the liionpack project. These benchmarks can be run using [airspeed velocity](https://asv.readthedocs.io/en/stable/) (`asv`).

### Running the benchmarks
First of all, you'll need `asv` installed:
```shell
pip install asv
```

To run the benchmarks for the latest commit on the `main` branch, simply enter the following command:
```shell
asv run
```
If it is the first time you run `asv`, you will be prompted for information about your machine (e.g. its name, operating system, architecture...).

Running the benchmarks can take a while, as all benchmarks are repeated several times to ensure statistically significant results. If accuracy isn't an issue, use the `--quick` option to avoid repeating each benchmark multiple times.
```shell
asv run --quick
```

Benchmarks can also be run over a range of commits. For instance, the following command runs the benchmark suite over every commit between a given commit with ID `commit_ID` and the tip of the `main` branch:
```shell
asv run commit_ID..develop
```
Further information on how to run benchmarks with `asv` can be found in the documentation at [Using airspeed velocity](https://asv.readthedocs.io/en/stable/using.html).

`asv` is configured using a file `asv.conf.json` located at the root of the liionpack repository. See the [asv reference](https://asv.readthedocs.io/en/stable/reference.html) for details on available settings and options.

Benchmark results are stored in a directory `results/` at the location of the configuration file. There is one result file per commit, per machine.

### Visualising benchmark results

`asv` is able to generate a static website with a visualisation of the benchmarks results, i.e. the benchmark's duration as a function of the commit hash.
To generate the website, use
```shell
asv publish
```
then, to view the website:
```shell
asv preview
```

Current benchmarks over liionpack's history can be viewed at https://pybamm-team.github.io/liionpack-bench/

### Adding benchmarks

To contribute benchmarks to liionpack, add a new benchmark function in one of the files in the `benchmarks/` directory.
Benchmarks are distributed across multiple files, grouped by theme. You're welcome to add a new file if none of your benchmarks fit into one of the already existing files.
Inside a benchmark file (e.g. `benchmarks/benchmarks.py`) benchmarks functions are grouped within classes.

Note that benchmark functions _must_ start with the prefix `time_`, for instance
```python3
def time_solve_model(self):
    BasicBenchmark.sim.solve([0, 1800])
```

In the case where some setup is necessary, but should not be timed, a `setup` function
can be defined as a method of the relevant class. For example:
```python3
class BasicBenchmark:
    def setup(self):
        self.sim = lp.basic_simulation()

    def time_solve_model(self):
        BasicBenchmark.sim.solve([0, 1800])
```

Similarly, a `teardown` method will be run after the benchmark. Note that, unless the `--quick` option is used, benchmarks are executed several times for accuracy, and both the `setup` and `teardown` function are executed before/after each repetition.

Running benchmarks can take a while, and by default encountered exceptions will not be shown. When developing benchmarks, it is often convenient to use the following command instead of `asv run`:
```shell
asv dev
```

`asv dev` implies options `--quick`, `--show-stderr`, and `--dry-run` (to avoid updating the `results` directory).


## Infrastructure

### Setuptools

Installation of liionpack _and dependencies_ is handled via [setuptools](http://setuptools.readthedocs.io/)

Configuration files:

```
setup.py
```

Note that this file must be kept in sync with the version number in `liionpack/__init__.py`.

### Continuous Integration using GitHub actions

Each change pushed to the liionpack GitHub repository will trigger the test and benchmark suites to be run, using [GitHub actions](https://github.com/features/actions).

Tests are run for different operating systems, and for all python versions officially supported by liionpack. If you opened a Pull Request, feedback is directly available on the corresponding page. If all tests pass, a green tick will be displayed next to the corresponding test run. If one or more test(s) fail, a red cross will be displayed instead.

Similarly, the benchmark suite is automatically run for the most recently pushed commit. Benchmark results are compared to the results available for the latest commit on the `develop` branch. Should any significant performance regression be found, a red cross will be displayed next to the benchmark run.

In all cases, more details can be obtained by clicking on a specific run.

Configuration files for various GitHub actions workflow can be found in `.github/worklfows`.

### Codecov

Code coverage (how much of our code is actually seen by the (linux) unit tests) is tested using [Codecov](https://docs.codecov.io/), a report is visible on https://codecov.io/gh/pybamm-team/liionpack.


### Read the Docs

Documentation is built using https://readthedocs.org/ and published on http://liionpack.readthedocs.io/.

### Google Colab

Editable notebooks are made available using [Google Colab](https://colab.research.google.com/) [here](https://colab.research.google.com/github/pybamm-team/liionpack/blob/main/).

### GitHub

GitHub does some magic with particular filenames. In particular:

- The first page people see when they go to [our GitHub page](https://github.com/pybamm-team/liionpack) displays the contents of our readme, which is written in the [Markdown](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet) format. Some guidelines can be found [here](https://help.github.com/articles/about-readmes/).
- The license for using liionpack is stored in [LICENSE](LICENSE), and [automatically](https://help.github.com/articles/adding-a-license-to-a-repository/) linked to by GitHub.
- This file, [contributing.md](contributing.md) is recognised as the contribution guidelines and a link is [automatically](https://github.com/blog/1184-contributing-guidelines) displayed when new issues or pull requests are created.

## Acknowledgements

This CONTRIBUTING.md file is largely based on the [PyBaMM](https://github.com/pybamm-team/PyBaMM/blob/develop/CONTRIBUTING.md) guidelines.
