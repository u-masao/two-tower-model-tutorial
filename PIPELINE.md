# pipeline DAG


## process DAG


```mermaid
flowchart TD
	node1["build_features"]
	node2["download_dataset"]
	node3["extract_dataset"]
	node4["pack_dataset"]
	node5["predict"]
	node6["train@benchmark"]
	node1-->node5
	node1-->node6
	node2-->node3
	node3-->node4
	node4-->node1
	node6-->node5
```


## output file DAG


```mermaid
flowchart TD
	node1["data/interim/dataset.cloudpickle"]
	node2["data/interim/features.cloudpickle"]
	node3["data/processed/predicted.parquet"]
	node4["data/raw/Book reviews/Book reviews/BX-Book-Ratings.csv"]
	node5["data/raw/Book reviews/Book reviews/BX-Users.csv"]
	node6["data/raw/Book reviews/Book reviews/BX_Books.csv"]
	node7["data/raw/Books Data with Category Language and Summary/Preprocessed_data.csv"]
	node8["data/raw/bookcrossing-dataset.zip"]
	node9["models/model.pth"]
	node1-->node2
	node2-->node3
	node2-->node9
	node4-->node1
	node5-->node1
	node6-->node1
	node8-->node4
	node8-->node5
	node8-->node6
	node8-->node7
	node9-->node3
```
