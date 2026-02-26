Place external holdout test CSV files in this folder if you want to keep them inside the project.

Example:
- Housing.csv

You can evaluate against an external file directly without copying by running:
python train.py --data-path data/housing.csv --test-data-path "C:\Users\chara\Downloads\archive (1)\Housing.csv" --model-path model.pkl
