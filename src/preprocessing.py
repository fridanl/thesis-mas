import pandas as pd
import re
from pathlib import Path

#################################### SENTIMENT #############################################
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

#################################### COMMONSENSE #############################################
# # loading the common sense data (parquet)
cs = pd.read_parquet("../data/commonsense/archive/train-00000-of-00001.parquet")
cs = cs[["question", "answer"]]
cs = cs.rename({"question": "text", "answer":"label"}, axis="columns")

# make id column
cs["id"] = cs.index
cs = cs[["id", "text", "label"]]
cs.to_csv("../data/commonsense/commonsense.csv", index=False)

#################################### SARCASM #############################################
txt_path = Path('data/sarc/raw/train_text.txt')
with txt_path.open(encoding="utf-8") as f:
    lines = [line.rstrip("\n") for line in f]

labels_path = Path('data/sarc/raw/train_labels.txt')
with labels_path.open(encoding='utf-8') as f:
    labels = [label.rstrip('\n') for label in f]

sarcasm = pd.DataFrame({'id': list(range(0, len(lines))),"text": lines, 'label': labels})

# count the number of words in each claim 
sarcasm['n_words'] = [len(x.split()) for x in sarcasm['text'].tolist()]

# filter out too short and too long examples 
sarcasm_pr = sarcasm[(sarcasm['n_words'] >= 6) & (sarcasm['n_words'] < 500)]

# remove examples that contain mentions of Israel, israeli, muslim, hillary, racism, racists, trump, russian, russia, slavery, jesus, gay, lesbian, homosexuality, communist, communism, terrorism, palestine, palestianians, rape, conservative, democrat, republican, antifa  
BASE_TERMS = [
    r"ukrain\w*", r"russi\w*", r"putin", r"zelensky\w*",
    r"iran\w*", r"iraq\w*", r"syria\w*", r"afghan\w*",
    r"gaza", r"west\s*bank", r"hamas", r"hezbollah", r"isis", r"taliban", r"idf",
    r"biden", r"obama", r"clinton", r"bush", r"reagan", r"sanders", r"aoc", r"desantis", r"pence", r"hillary",
    r"democrat\w*", r"republican\w*", r"liberal\w*", r"progressive\w*", r"conservativ\w*",
    r"libertarian\w*", r"socialist\w*", r"marxist\w*", r"fascist\w*", r"nazi\w*", r"alt-?right", r"woke",
    r"fascism", r"communism", r"socialism", r"liberalism", r"conservatism", r"antifa",
    r"islam\w*", r"muslim\w*", r"christian\w*", r"catholic\w*", r"jewish", r"judais\w*", r"zionis\w", r"israel", r"palestine", r"palestinians",
    r"hindu\w*", r"sikh\w*", r"buddhist\w*", r"buddhis\w*", r"atheis\w*",
    r"trans(?:gender)?\w*", 
    r"non-?binary", r"lgbt\w*", r"queer", r"bisexual\w*", r"pansexual\w*", r"pronoun\w*",
    r"immigra\w*", r"migrant\w*", r"refugee\w*", r"abortion", r"pro-?life", r"pro-?choice", r"suicide"
    r"gun\w*", r"firearm\w*", r"nra", r"racist\w",
    r"covid\w*", r"coronavirus", r"pandemic", r"lockdown\w*", r"vaccine\w*", r"anti-?vax\w*",
    r"hitler", r"stalin", r"genocide\w*", r"holocaust", r"apartheid",
    r"climate\s*change", r"global\s*warming", r"feminis\w*", r"#metoo",
]

pattern = re.compile(r"\b(?:%s)\b" % "|".join(BASE_TERMS), flags=re.IGNORECASE)
mask_has_banned = sarcasm_pr["text"].str.contains(pattern, na=False)

################ this is the dataset we call sarcasm.csv ################### 
sarcasm_clean = sarcasm_pr[~mask_has_banned].copy()
flagged = sarcasm_pr[mask_has_banned].copy()

