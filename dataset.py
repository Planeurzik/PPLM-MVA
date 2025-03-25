from datasets import load_dataset
from joblib import Memory
import re
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

def print_n_entries(output_file):
    # use name="sample-10BT" to use the 10BT sample
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train",
        streaming=True)


    with open(output_file, "w", encoding="utf-8") as f:
        # Iterate through the dataset
        for entry in iter(fw):
            if entry["language"]=="en":
                f.write(entry["text"])



print_n_entries("output.txt")
