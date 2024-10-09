# pipeline DAG


## process DAG


```mermaid
flowchart TD
	node1["build_features"]
	node2["download_dataset"]
	node3["extract_dataset"]
	node4["pack_dataset"]
	node2-->node3
	node3-->node4
	node4-->node1
```


## output file DAG


```mermaid
flowchart TD
	node1["data/interim/dataset.cloudpickle"]
	node2["data/interim/features.cloudpickle"]
	node3["data/raw/Book reviews/Book reviews/BX-Book-Ratings.csv"]
	node4["data/raw/Book reviews/Book reviews/BX-Users.csv"]
	node5["data/raw/Book reviews/Book reviews/BX_Books.csv"]
	node6["data/raw/Books Data with Category Language and Summary/Preprocessed_data.csv"]
	node7["data/raw/bookcrossing-dataset.zip"]
	node1-->node2
	node3-->node1
	node4-->node1
	node5-->node1
	node7-->node3
	node7-->node4
	node7-->node5
	node7-->node6
```
