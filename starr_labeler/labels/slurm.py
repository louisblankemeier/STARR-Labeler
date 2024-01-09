from python_submit import python_submit

python_submit("python main.py --config-name afib.yaml", node="roma")

python_submit("python main.py --config-name heart_failure.yaml", node="roma")

python_submit("python main.py --config-name cardiovascular_disease.yaml", node="roma")

python_submit(
    "python main.py --config-name cardiovascular_disease_afib_heart_failure.yaml",
    node="roma",
)
