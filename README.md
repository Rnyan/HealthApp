# HealthApp

The main purpose of this app is to help people eat healthier, and this app does it through providing recipes that fit the user's needs.

## Description

The current app is a small prototype where I am testing how TensorFlow works. Currently, I only have a model that is trained and then used to predict the best "recipe" based on hard coded values. The next goal is to build a prototype flutter app and then later use a recipe API for the AI to find recipes on and turn my current model into an API for flutter to send and pull data from.

## Getting Started

### Dependencies

* Windows 10 or higher
* Python 3.11 or 3.13
* TensorFlow
* pandas
* joblib
* scikit-learn

### Installing
1. Install Python 3.11 or 3.13 from [python.org](https://www.python.org/downloads/).
2. Download the project from this repository, or clone it:

```powershell
git clone https://github.com/Rnyan/HealthApp.git
cd HealthApp
```
3. Set the path to the project folder:
```powershell
cd "C:\path\to\NameOfProjectFolder"
```

4. Create and activate a virtual environment:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

5. Install the required dependencies:

```powershell
python -m pip install -r requirements.txt
```

6. `recipes.csv` and `restrictions.csv` should be in the project folder.

The `models` folder and trained model files are created when `train.py` is run.

### Executing program

* Install the dependencies through the requirements file:
```powershell
python -m pip install -r requirements.txt
```
* Run train.py to train the model
```powershell
python train.py
```
* Have the model predict the best recipe
```
python predict.py
```
The output should print a recipe id, a category, and the confidence level.

## Authors
— [@Rnyan](https://github.com/Rnyan)

## Version History
* 0.2
    * Model prediction and recipe filtering
* 0.1
    * Initial recipe dataset and model training prototype

## License

This project is licensed under the MIT License - see the LICENSE.md file for details

## Acknowledgments

- TensorFlow site: https://www.tensorflow.org/
- pandas site: https://pandas.pydata.org/docs/
- scikit-learn site: https://scikit-learn.org/stable/
- Python site: https://docs.python.org/3/

AI tools were used for guidance and assistance on the project.
