import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

# loading the sentoment data
text_path = "../data/sentiment/archive/train_text.txt"
val_text_path = "../data/sentiment/archive/val_text.txt"
test_text_path = "../data/sentiment/archive/test_text.txt"
labels_path = "../data/sentiment/archive/train_labels.txt"
val_labels_path = "../data/sentiment/archive/val_labels.txt"
test_labels_path = "../data/sentiment/archive/test_labels.txt"

#train
def load_text(text_path):
    with open(text_path, "r", encoding="utf-8") as f:
        text = pd.DataFrame(
            [line.rstrip("\n") for line in f],
            columns=["text"]
        )
    return text

def load_labels(labels_path):
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = pd.DataFrame(
            [line.rstrip("\n") for line in f],
            columns=["label"]
        )
    return labels

def preprocess(text, labels):
    text["id"] = text.index
    text = text[["id", "text"]]
    labels["id"] = labels.index
    text["label"] = list(labels["label"])
    text.drop(columns="id") # drop the id
    return text


#load text    
train_text = load_text(text_path=text_path)
val_text = load_text(text_path=val_text_path)
test_text = load_text(text_path=test_text_path)
#load labels  
train_labels = load_labels(labels_path=labels_path)
val_labels = load_labels(labels_path=val_labels_path)
test_labels = load_labels(labels_path=test_labels_path)

train = preprocess(train_text, train_labels)
val = preprocess(val_text, val_labels)
test = preprocess(test_text, test_labels)

# Merging train, val, test 
print(f'extected length of data: {len(train) + len(test) + len(val)}')
data = pd.concat([train, val, test])
print(f'actual length of data: {len(data)}')

# Clean data 
def clean_filter(df):
    if re.fullmatch(r"\W+", df): # remove pure special characters like "("
        return False
    
    if re.fullmatch(r"\d+", df): # removes pure numbers
        return False
    
    return True


mask = data["text"].apply(clean_filter)
clean_data = data[mask]          # kept rows
removed_text = data[~mask]       # removed rows

print(f'number of removed rows are {len(removed_text)}')
### Removing duplicates and semi-duplicates

# Many of the sentences are near duplicates, since the dataset is a treebank.

# We want to remove the semi-duplicates if the labels are identical, but keep them, if the interposed sentence changes the sentiment, and therefore the label.

# **Example:**

#     25861,"The film boasts at least a few good ideas and features some decent performances ,",positive

#     25862,"The film boasts at least a few good ideas and features some decent performances , but",positive

#     25863,"The film boasts at least a few good ideas and features some decent performances , but the result is disappointing",negative

# Here, if there are semi-duplicates, we remove the shortest one. So the first sentence will be removed in this case.

def normalise_text(s):
    s = s.lower()  # lowercase
    s = s.replace("``", '"').replace("''", '"')  # norm the quotations
    s = re.sub(r'\s+', ' ', s) # normalise the spaces 
    s = s.strip()
    return s

def remove_semi_duplicates(df):
    """
    Removes consecutive semi-duplicate sentences that are strict prefixes of the next sentence
    if labels are the same. Keeps all sentences if labels differ.
    """
    keep_indices = []
    n = len(df)
    
    for i in range(n):
        keep = True
        # the dataset is already sorted, so only look at the next ones, and not the entire dataset.
        if i < n - 1:
            t1, l1 = normalise_text(df.iloc[i]["text"]), df.iloc[i]["label"]
            t2, l2 = normalise_text(df.iloc[i+1]["text"]), df.iloc[i+1]["label"]
            
            if t2.startswith(t1) and l1 == l2:
                # current sentence is a prefix of the next, same label -> drop current
                keep = False
        
        if keep:
            keep_indices.append(df.index[i])
    
    return df.loc[keep_indices].reset_index(drop=True)




df = remove_semi_duplicates(clean_data)
print(f'the length of df before: {len(clean_data)}, the length now: {len(df)}')
print(f'removed: {len(clean_data)-len(df)}')
df["id"] = df.index # reset the id (that we dropped before)
df = df[["id", "text", "label"]] # rearrange the columns
df.head()


df.to_csv("../data/sentiment/sentiment.csv", index=False)




# Commonsense
# # loading the common sense data (parquet)
cs = pd.read_parquet("../data/commonsense/archive/train-00000-of-00001.parquet")
cs = cs[["question", "answer"]]
cs = cs.rename({"question": "text", "answer":"label"}, axis="columns")

# make id column
cs["id"] = cs.index
cs = cs[["id", "text", "label"]]
cs.to_csv("../data/commonsense/commonsense.csv", index=False)


sns.displot(text, x="label");


text["sentence_length"] = text["text"].str.split().apply(len)
plt.figure(figsize=(8, 5))

sns.histplot(
    text["sentence_length"],
    bins=50,
    kde=True
)

plt.xlabel("Sentence Length (words)")
plt.ylabel("Frequency")
plt.title("Distribution of Sentence Lengths")

plt.show()
