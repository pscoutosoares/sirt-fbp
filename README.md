## Base de dados
1. Download the dataset at https://zenodo.org/record/3384092#.X4BudnVKhhE

2. Then, extract all zip files at one same directory

## Enviroment
Make sure to have anaconda installed and import enviroment from condaEnviroment.yml file. 

Just use the command `conda env create --file envname.yml`

## Run
1. Open sirtfbp.py file and set PATH variable with dataset location
2. Run sirtfbp.py (The first time will take longer, because of the iterative filter creation)
3. Set PATH variable at compare_quality.py file and run. The result going to be displayed at console as a log.
