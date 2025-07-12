all: 
	python src/data_fit.py

process:
	python src/data_process.py
visualize:
	python src/data_visualization.py
e:
	python src/exp.py
r:
	git restore .
clean:
	rm -rf src/visualizations/*
missing:
	python src/data_missing.py
res:
	git restore config/feature.json
ptest:
	python src/data_process_test.py