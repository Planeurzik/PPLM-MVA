from datasets import load_dataset
from joblib import Memory
# Create a Memory object with a specified cache directory
memory = Memory("cache_dir", verbose=0)
@memory.cache
def get_n_entries(n):

    # use name="sample-10BT" to use the 10BT sample
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train",
        streaming=True)

    # Initialize a list to store the last 10entries
    last_10_entries = []

    # Iterate through the dataset
    for entry in iter(fw):
        last_10_entries.append(entry)
        if len(last_10_entries)>n:
            break
    return last_10_entries

last_10_entries = get_n_entries(10000)


dataset_en = []
# Print the last 10 entries
for entry in last_10_entries:
    if entry["language"]=="en":
       dataset_en.append(entry) 
print(len(last_10_entries), len(dataset_en))
