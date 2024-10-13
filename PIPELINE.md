# pipeline DAG


## process DAG


```mermaid
flowchart TD
	node1["build_features"]
	node2["download_dataset"]
	node3["evaluate@benchmark"]
	node4["evaluate@setup_1"]
	node5["evaluate@setup_2"]
	node6["extract_dataset"]
	node7["pack_dataset"]
	node8["predict@benchmark"]
	node9["predict@setup_1"]
	node10["predict@setup_2"]
	node11["train@benchmark"]
	node12["train@setup_1"]
	node13["train@setup_2"]
	node1-->node8
	node1-->node9
	node1-->node10
	node1-->node11
	node1-->node12
	node1-->node13
	node2-->node6
	node6-->node7
	node7-->node1
	node8-->node3
	node9-->node4
	node10-->node5
	node11-->node8
	node12-->node9
	node13-->node10
```


## output file DAG


```mermaid
flowchart TD
	node1["data/interim/dataset.cloudpickle"]
	node2["data/interim/features.cloudpickle"]
	node3["data/processed/predicted_benchmark.parquet"]
	node4["data/processed/predicted_setup_1.parquet"]
	node5["data/processed/predicted_setup_2.parquet"]
	node6["data/raw/Book reviews/Book reviews/BX-Book-Ratings.csv"]
	node7["data/raw/Book reviews/Book reviews/BX-Users.csv"]
	node8["data/raw/Book reviews/Book reviews/BX_Books.csv"]
	node9["data/raw/Books Data with Category Language and Summary/Preprocessed_data.csv"]
	node10["data/raw/bookcrossing-dataset.zip"]
	node11["models/model_benchmark.pth"]
	node12["models/model_setup_1.pth"]
	node13["models/model_setup_2.pth"]
	node14["reports/figures/evaluate-benchmark"]
	node15["reports/figures/evaluate-setup_1"]
	node16["reports/figures/evaluate-setup_2"]
	node1-->node2
	node2-->node3
	node2-->node4
	node2-->node5
	node2-->node11
	node2-->node12
	node2-->node13
	node3-->node14
	node4-->node15
	node5-->node16
	node6-->node1
	node7-->node1
	node8-->node1
	node10-->node6
	node10-->node7
	node10-->node8
	node10-->node9
	node11-->node3
	node12-->node4
	node13-->node5
```
