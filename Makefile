install:
	pipenv install --dev

train: install
	cd data && pipenv run python add_bg_noise.py && cd .. && pipenv run python lstm.py
