.PHONY: install clean-data eda train test lint api docker-build docker-run

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

clean-data:
	python -m src.data --input data/raw/processed.cleveland.data --output data/processed/heart_disease_clean.csv

eda:
	python -m src.eda

train:
	python -m src.train

test:
	pytest

lint:
	ruff check .

api:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t heart-disease-api:1.0.0 .

docker-run:
	docker run --rm -p 8000:8000 heart-disease-api:1.0.0
