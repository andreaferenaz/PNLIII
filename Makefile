.PHONY: install index test eval images eval-vision ejemplos api mock ui

install:
	pip install -r requirements.txt && pip install -e .

index:
	python scripts/build_index.py

test:
	LLM_PROVIDER=mock VISION_BACKEND=mock pytest -q

eval:
	python scripts/evaluate.py

images:
	python scripts/fetch_eval_images.py --n 100

eval-vision:
	python scripts/evaluate_vision.py

ejemplos:
	python scripts/ejemplos.py

api:
	uvicorn eppagent.api.main:app --reload --port 8000

mock:
	LLM_PROVIDER=mock VISION_BACKEND=mock uvicorn eppagent.api.main:app --reload --port 8000

ui:
	streamlit run app_streamlit.py --server.port 8501
