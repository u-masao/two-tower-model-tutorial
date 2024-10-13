# pipeline DAG


## process DAG


```mermaid
flowchart TD
	node1["build_features"]
	node2["download_dataset"]
	node3["extract_dataset"]
	node4["pack_dataset"]
	node5["predict@benchmark"]
	node6["predict@setup_1"]
	node7["train@benchmark"]
	node8["train@setup_1"]
	node1-->node5
	node1-->node6
	node1-->node7
	node1-->node8
	node2-->node3
	node3-->node4
	node4-->node1
	node7-->node5
	node8-->node6
```


## output file DAG


```mermaid
flowchart TD
	node1["data/interim/dataset.cloudpickle"]
	node2["data/interim/features.cloudpickle"]
	node3["data/processed/predicted_benchmark.parquet"]
	node4["data/processed/predicted_setup_1.parquet"]
	node5["data/raw/Book reviews/Book reviews/BX-Book-Ratings.csv"]
	node6["data/raw/Book reviews/Book reviews/BX-Users.csv"]
	node7["data/raw/Book reviews/Book reviews/BX_Books.csv"]
	node8["data/raw/Books Data with Category Language and Summary/Preprocessed_data.csv"]
	node9["data/raw/bookcrossing-dataset.zip"]
	node10["models/model_benchmark.pth"]
	node11["models/model_setup_1.pth"]
	node1-->node2
	node2-->node3
	node2-->node4
	node2-->node10
	node2-->node11
	node5-->node1
	node6-->node1
	node7-->node1
	node9-->node5
	node9-->node6
	node9-->node7
	node9-->node8
	node10-->node3
	node11-->node4
```
