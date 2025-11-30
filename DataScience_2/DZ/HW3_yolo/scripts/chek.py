from datasets import load_dataset

ds = load_dataset("cj-mills/hagrid-sample-30k-384p", split="train")

# Посмотрим названия колонок
print("Column names:", ds.column_names)

# Посмотрим первый элемент датасета
print("First example:", ds[0])
