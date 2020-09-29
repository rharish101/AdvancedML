# Advanced Machine Learning

This is the repository for the projects of the course "Advanced Machine Learning" (252-0535-00L), offered in Autumn 2020 at ETH Zurich.

## Group

The group name is "Advanced Machines".
The group members are:

* András Geiszl (ageiszl@student.ethz.ch)
* Daniël Trujillo (dtrujillo@student.ethz.ch)
* Harish Rajagopal (hrajagopal@student.ethz.ch)

### For Contributors

1. Install [Poetry](https://github.com/python-poetry/poetry)
1. Clone project and change into the project directory
1. Create a Python virtual environment (e.g. `conda create -n aml python=3.8.5` or `virtualenv env`)
1. Activate the environment (e.g. `conda activate aml` or `source ./env/bin/activate`)
1. Install required dependencies
    * Development: `poetry install`
    * Deployment: `poetry install --no-dev`
1. Install pre-commit hooks from the root directory of this repository:

    ```sh
    pre-commit install
    ```

    **NOTE**: You need to be inside the virtual environment every time you commit.
1. Install the [Pylance language server](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance) for VSCode (or if you're using a different code editor, install [Pyright](https://github.com/microsoft/pyright))
