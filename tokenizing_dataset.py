from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders
from utils import proper_decode

# Define the BPE tokenizer
tokenizer = Tokenizer(models.BPE())

# Configure pre-tokenization to split words efficiently
#tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

tokenizer.decoder = decoders.ByteLevel()

tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

# Setup a trainer to learn from the dataset
trainer = trainers.BpeTrainer(special_tokens=["<s>", "</s>", "<unk>", "<pad>"], vocab_size=10_000)

# Load dataset
with open("nice_big.txt", "r", encoding="utf-8") as f:
    text_data = f.readlines()

# Train tokenizer on the text dataset
tokenizer.train_from_iterator(text_data, trainer)

# Save tokenizer for future use
tokenizer.save("bpe_tokenizer.json")

encoded = tokenizer.encode("""This is a sample sentence.
                           this is madness
                           """)
# Example usage: Encoding a sentence
print("Tokens:", encoded.tokens)
print("Token IDs:", encoded.ids)
print(tokenizer.decode(encoded.ids))
#print("Decoded:", proper_decode(encoded.tokens))