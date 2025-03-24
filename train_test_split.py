f = open("dataset/nice_output2.txt","r")
dataset = f.read()
f.close()

def train_test_split_text(text, split_ratio=0.8):
    split_point = int(len(text) * split_ratio)
    train_text = text[:split_point]
    test_text = text[split_point:]
    return train_text, test_text

train_dataset, test_dataset = train_test_split_text(dataset, split_ratio=0.9)
f = open("dataset/train.txt","w")
f.write(train_dataset)
f.close()
f = open("dataset/test.txt","w")
f.write(test_dataset)
f.close()