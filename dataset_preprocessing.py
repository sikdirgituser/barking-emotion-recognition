import pandas as pd
import matplotlib.pyplot as plt

# Load tables from the Google Drive folder
tabs = ['Kathi', 'Luca', 'Menghan', 'Pascal']
sheet_id = "1MFeAULgteVBXRxSgZZ5z7IDsWI8cSD36zG4DOhfvCVs"
df_labels = pd.DataFrame()

# download single tabs & merge to one dataframe
for tab in tabs:
    url = "http://docs.google.com/spreadsheets/d/{}/gviz/tq?tqx=out:csv&sheet={}".format(sheet_id, tab)
    print('read {}'.format(url))
    df = pd.read_csv(url).iloc[:, 0:6]
    df.loc[:, 'labeled_by'] = tab
    df_labels = pd.concat([df_labels, df], ignore_index=True)

# drop unlabeled entries
df_labels.dropna(subset=['emotion_label_1'], inplace=True)

# prepare emotion categories
labels = list(df_labels.emotion_label_1.unique())
CATEGORIES = [
    'Positive',
    'Positive',
    'Negative',
    'Negative',
    'Neutral',
    'Neutral',
    'Negative',
    'Neutral',
    'Neutral',
]

def is_same_category(labs):
    categorized = [CATEGORIES[labels.index(label)] for label in labs]
    return ' | '.join(categorized)

# group by audioset_index and check if labeled the same by both labelers
df_agg = df_labels.groupby('audioset_index').agg(
    count=('emotion_label_1', lambda x: len(x)),
    label_same=('emotion_label_1', lambda labs: list(labs)[0] == list(labs)[1] if len(labs) > 1 else False),
    label=('emotion_label_1', lambda labs: ' | '.join(labs)),
    labels_category=('emotion_label_1', lambda labs: is_same_category(labs)),
    ytid=('ytid', 'last'),
    start=('start', 'last'),
    stop=('stop', 'last'),
)
df_agg['same_category'] = df_agg.labels_category.apply(
    lambda x: x.split(' | ')[0] == x.split(' | ')[1] if len(x.split('|')) > 1 else False)

# subgroup for same category
df_categories_same = df_agg[df_agg.same_category == True].copy()
df_categories_same.loc[:, 'category'] = df_categories_same['labels_category'].apply(lambda x: x.split(' | ')[0])

# only valid labels
df_labels_same = df_agg[df_agg.label_same == True]
df_labels_same.loc[:, 'label'] = df_labels_same['label'].apply(lambda x: x.split(' | ')[0])

# add pre-labelled data from special tab
url = "http://docs.google.com/spreadsheets/d/{}/gviz/tq?tqx=out:csv&sheet={}".format(sheet_id, "pre-labelled")
print('read {}'.format(url))
df_pre = pd.read_csv(url).iloc[:, 0:8]
df_pre.rename(columns={
    'emotion_label_1': 'label',
    'emotion_label_2': 'label_2'
}, inplace=True)
df_labels_same = pd.concat([df_labels_same, df_pre[['label', 'label_2', 'ytid', 'start', 'stop']]])

# plot emotion category count
plt.subplots(figsize=(12, 8))
df_labels_same['label'].value_counts().plot(kind='barh', title=f'Examples per category (Total: {df_labels_same.shape[0]})')
plt.show()

# export as csv
df_labels_same[['label', 'ytid', 'start', 'stop']].to_csv('data/dataset.csv')


print(df_labels_same['label'].value_counts())


