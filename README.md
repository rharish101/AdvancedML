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
1. For VSCode install
    * [Python Extension Pack](https://marketplace.visualstudio.com/items?itemName=donjayamanne.python-extension-pack) (`donjayamanne.python-extension-pack`)
    * [Pylance language server](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance) (`ms-python.vscode-pylance`)
1. For any other editors install
    * Their own set of Python languge services
    * [Pyright](https://github.com/microsoft/pyright)
1. Clone project and change into the project directory
1. If you're using VSCode, copy `.vscode/settings.json.default` to `.vscode/settings.json`.
1. Create a Python virtual environment (e.g. `conda create -n aml python=3.8.5` or `virtualenv env`)
1. In VSCode, Press `Ctrl + Shift + P` and select `Python: Select Interpreter` to set your virtual environment as the default for this workspace (you might need to restart VSCode to see the desired option or specify the path manually).
1. Activate the environment (e.g. `conda activate aml` or `source ./env/bin/activate`)
1. Install required dependencies with `poetry install`
1. Install pre-commit hooks from the root directory of this repository:

    ```sh
    pre-commit install
    ```

**NOTE**: You need to be inside the virtual environment every time you commit.
