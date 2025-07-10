all:
	python src/data_process.py
process:
	python src/data_processing.py
visualize:
	python src/data_visualization.py
e:
	python src/exp.py

r:
	git restore .
clean:
	rm -rf src/visualizations/*