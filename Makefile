.PHONY: install test backtest scorelines report app all

install:
	pip install -e ".[app,dev]"

test:
	python -m pytest -q

backtest:
	python -m plpred.backtest

scorelines:
	python -m plpred.scorelines

report:
	python -m plpred.report

app:
	streamlit run app/streamlit_app.py

all: test backtest scorelines report
