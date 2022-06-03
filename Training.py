#%%
import re
import string
import numpy as np
import pandas as pd
import seaborn as sns
from bs4 import BeautifulSoup
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from spacy.lang.en.stop_words import STOP_WORDS as stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Flatten, Embedding, Dropout, Conv1D, BatchNormalization, Activation, MaxPooling1D, GlobalMaxPooling1D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model
#suppressing the warnings
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
import warnings
warnings.filterwarnings('ignore')
plt.style.use('ggplot')
# %%
dataset = pd.read_csv('C:/Users/GAURAV/Desktop/MajorProject/.vscode/Depression_Detection.csv')
dataset.rename(columns={'text':'Text', 'class':'Class'}, inplace=True)
dataset.sample(5)
# %%
dataset.loc[dataset['Class'] == 'teenagers','Class'] = 'Other'
dataset.loc[dataset['Class'] == 'depression','Class'] = 'Depression'
dataset.sample(5)

# %%
print(f'Size Of Dataset: {dataset.shape}')
# %%
plt.rcParams['figure.figsize'] = [8,5.5]
plt.rcParams['figure.dpi'] = 90
sns.set(style='darkgrid')

sns.countplot('Class', data=dataset)
plt.ylabel("Number of Samples")
plt.title('Distribution Of Target Variable', pad=20)
plt.show()
# %%
dataset['Class'].value_counts().plot.pie(autopct='%1.2f%%', shadow=True)
plt.title('Distribution Of Target Variable In Percentage')
plt.ylabel("")
plt.show()
# %%
def char_counts(x):
  #spliting the words
  s = x.split()
  #joining without space
  x = ''.join(x)
  return len(x)
# %%
#adding more information for exploratory data analysis
dataset['Word_Counts'] = dataset['Text'].apply(lambda x: len(str(x).split()))
dataset['Char_Counts'] = dataset['Text'].apply(lambda x: char_counts(str(x)))
dataset['Stop_Words_Count'] = dataset['Text'].apply(lambda x: len([word for word in str(x).split() if word in stopwords]))
dataset['Unique_Word_Count'] = dataset['Text'].apply(lambda x: len(set(str(x).split())))
dataset['Punctuation_Count'] = dataset['Text'].apply(lambda x: len([word for word in str(x) if word in string.punctuation]))
# %%
plt.rcParams['figure.figsize'] = [10,5]
plt.rcParams['figure.dpi'] = 95
# %%
ig, ax = plt.subplots()
sns.kdeplot(dataset[dataset['Class']=='Depression']['Word_Counts'], label='Depression', color='red', shade=True, x=1000)
sns.kdeplot(dataset[dataset['Class']=='SuicideWatch']['Word_Counts'], label='Suicide', color='darkblue', shade=True, x=1000)
sns.kdeplot(dataset[dataset['Class']=='Other']['Word_Counts'], label='Other', color='green', shade=True, x=1000)
ax.set_xlabel('Word Count')    
plt.title('Word Count Density Plot', pad=10,  fontsize=15)
plt.legend(loc = 'upper right', prop={'size': 9.5})

ax2 = plt.axes([0.2, 0.4, .2, .2], facecolor='y')
sns.kdeplot(dataset[dataset['Class']=='Depression']['Word_Counts'],  color='red', shade=True)
sns.kdeplot(dataset[dataset['Class']=='SuicideWatch']['Word_Counts'], color='darkblue', shade=True)
sns.kdeplot(dataset[dataset['Class']=='Other']['Word_Counts'], color='green', shade=True)
ax2.axes.get_xaxis().set_ticks([])
ax2.axes.get_yaxis().set_ticks([])
plt.xlabel('Word Count', fontsize=8)
plt.ylabel('Density', fontsize=8)
ax2.set_title('zoom')
ax2.set_xlim([0,500])
ax2.set_ylim([0,0.0075])
plt.show()
# %%
fig, ax = plt.subplots()
sns.kdeplot(dataset[dataset['Class']=='Depression']['Char_Counts'], label='Depression', color='red', shade=True, x=1000)
sns.kdeplot(dataset[dataset['Class']=='SuicideWatch']['Char_Counts'], label='Suicide', color='darkblue', shade=True, x=1000)
ax.set_xlabel('Character Count')    
plt.title('Character Count Density Plot', pad=10,  fontsize=15)
plt.legend(loc = 'upper right', prop={'size': 9.5})

ax2 = plt.axes([0.2, 0.4, .2, .2], facecolor='y')
sns.kdeplot(dataset[dataset['Class']=='Depression']['Char_Counts'],  color='red', shade=True)
sns.kdeplot(dataset[dataset['Class']=='SuicideWatch']['Char_Counts'], color='darkblue', shade=True)
ax2.axes.get_xaxis().set_ticks([])
ax2.axes.get_yaxis().set_ticks([])
plt.xlabel('Character Count', fontsize=8)
plt.ylabel('Density', fontsize=8)
ax2.set_title('zoom')
ax2.set_xlim([0,500])
ax2.set_ylim([0.0005,0.00095])
plt.show()
# %%
fig, ax = plt.subplots()
sns.kdeplot(dataset[dataset['Class']=='Depression']['Stop_Words_Count'], label='Depression', color='red', shade=True, x=1000)
sns.kdeplot(dataset[dataset['Class']=='SuicideWatch']['Stop_Words_Count'], label='Suicide', color='darkblue', shade=True, x=1000)
ax.set_xlabel('Stop Word Count')    
plt.title('Stop Word Count Density Plot', pad=10,  fontsize=15)
plt.legend(loc = 'upper right', prop={'size': 9.5})

ax2 = plt.axes([0.2, 0.4, .2, .2], facecolor='y')
sns.kdeplot(dataset[dataset['Class']=='Depression']['Stop_Words_Count'],  color='red', shade=True)
sns.kdeplot(dataset[dataset['Class']=='SuicideWatch']['Stop_Words_Count'], color='darkblue', shade=True)
sns.kdeplot(dataset[dataset['Class']=='Other']['Stop_Words_Count'], color='green', shade=True)
ax2.axes.get_xaxis().set_ticks([])
ax2.axes.get_yaxis().set_ticks([])
plt.xlabel('Stop Word Count', fontsize=8)
plt.ylabel('Density', fontsize=8)
ax2.set_title('zoom')
ax2.set_xlim([0,500])
ax2.set_ylim([0,0.011])
plt.show()
# %%
fig, ax = plt.subplots()
sns.kdeplot(dataset[dataset['Class']=='Depression']['Unique_Word_Count'], label='Depression', color='red', shade=True, x=1000)
sns.kdeplot(dataset[dataset['Class']=='SuicideWatch']['Unique_Word_Count'], label='Suicide', color='darkblue', shade=True, x=1000)
sns.kdeplot(dataset[dataset['Class']=='Other']['Unique_Word_Count'], label='Other', color='green', shade=True, x=1000)
ax.set_xlabel('Unique Word Count')    
plt.title('Unique Word Count Density Plot', pad=10,  fontsize=15)
plt.legend(loc = 'upper right', prop={'size': 9.5})

ax2 = plt.axes([0.2, 0.4, .2, .2], facecolor='y')
sns.kdeplot(dataset[dataset['Class']=='Depression']['Unique_Word_Count'],  color='red', shade=True)
sns.kdeplot(dataset[dataset['Class']=='SuicideWatch']['Unique_Word_Count'], color='darkblue', shade=True)
sns.kdeplot(dataset[dataset['Class']=='Other']['Unique_Word_Count'], color='green', shade=True)
ax2.axes.get_xaxis().set_ticks([])
ax2.axes.get_yaxis().set_ticks([])
plt.xlabel('Unique Word Count', fontsize=8)
plt.ylabel('Density', fontsize=8)
ax2.set_title('zoom')
ax2.set_xlim([0,500])
ax2.set_ylim([0,0.0075])
plt.show()
# %%
fig, ax = plt.subplots()
sns.kdeplot(dataset[dataset['Class']=='Depression']['Punctuation_Count'], label='Depression', color='red', shade=True, x=1000)
sns.kdeplot(dataset[dataset['Class']=='SuicideWatch']['Punctuation_Count'], label='Suicide', color='darkblue', shade=True, x=1000)
sns.kdeplot(dataset[dataset['Class']=='Other']['Punctuation_Count'], label='Other', color='green', shade=True, x=1000)
ax.set_xlabel('Punctuation Word Count')    
plt.title('Punctuation Word Count Density Plot', pad=10,  fontsize=15)
plt.legend(loc = 'upper right', prop={'size': 9.5})

ax2 = plt.axes([0.2, 0.4, .2, .2], facecolor='y')
sns.kdeplot(dataset[dataset['Class']=='Depression']['Punctuation_Count'],  color='red', shade=True)
sns.kdeplot(dataset[dataset['Class']=='SuicideWatch']['Punctuation_Count'], color='darkblue', shade=True)
sns.kdeplot(dataset[dataset['Class']=='Other']['Punctuation_Count'], color='green', shade=True)
ax2.axes.get_xaxis().set_ticks([])
ax2.axes.get_yaxis().set_ticks([])
plt.xlabel('Punctuation Word Count', fontsize=8)
plt.ylabel('Density', fontsize=8)
ax2.set_title('zoom')
ax2.set_xlim([0,500])
ax2.set_ylim([0,0.0075])
plt.show()
# %%
import contractions
import unicodedata
# %%
def remove_accented(x):
  x = unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8', 'ignore')
  return x
# %%
freq_comm = pd.Series(dataset['Text']).value_counts()
f20 = freq_comm[:50]
rare20 = freq_comm.tail(50)
# %%
def get_clean(X):
  X = str(X).lower()
  X = X.replace('\\', ' ').replace('_', ' ').replace('.', ' ').replace(':', '')
  X = re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?',"",  X)
  X = re.sub(r'\brt\b', '', X).strip()
  X = re.sub('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});',"", X)
  X = re.sub(r'[^\w\d\s]+','', X)
  X = ' '.join(X.split())
  X = BeautifulSoup(X, 'lxml').get_text().strip()
  X = remove_accented(X)
  X = re.sub(r'[^\w ]+','',X)
  X = re.sub("(.)\\1{2,}", "\\1", X)
  X = contractions.fix(X)
  #X = ' '.join([word for word  in X.split() if word not in stopwords])
  X = ' '.join([word for word in X.split() if word not in f20]) 
  X = ' '.join([word for word in X.split() if word not in rare20])
  return X
# %%
dataset['Text'] = dataset['Text'].apply(lambda X : get_clean(X))
dataset = dataset[dataset['Text'].str.split().str.len().ge(3)]
# %%
dataset.sample(4)
# %%
toxic = str(dataset[dataset['Class']=='Depression'].Text)
word_cloud = WordCloud(width=550, height=500, max_font_size=150).generate(toxic)
plt.imshow(word_cloud)
plt.title('Word Cloud Representation For Text In Depression Category')
plt.axis('off')
plt.show()
# %%
toxic = str(dataset[dataset['Class']=='SuicideWatch'].Text)
word_cloud = WordCloud(width=550, height=500, max_font_size=150).generate(toxic)
plt.imshow(word_cloud)
plt.title('Word Cloud Representation For Text In SuicideWatch Category')
plt.axis('off')
plt.show()

# %%
X = dataset['Text']
# %%
Y_df = dataset['Class']
Y_df = pd.DataFrame({'Class':Y_df.values})
Y_df['Class'] = Y_df.Class.astype('category')
Y_df['Label_Code'] = Y_df["Class"].cat.codes
display(Y_df.sample(5))
# %%
Y = to_categorical(Y_df['Label_Code'])
Y = pd.DataFrame({'Depression': Y[:, 0], 'SuicideWatch': Y[:, 1], 'Other': Y[:, 2]})
display(Y.sample(4))
# %%
token = Tokenizer()
token.fit_on_texts(X)
# %%
vocab_size = len(token.word_index) + 1
print('The size of vocab:', vocab_size)
#%%
encoded_text = token.texts_to_sequences(X)
max_length = len(X.max()) + 5  
X = pad_sequences(encoded_text, maxlen=max_length, padding='pre')
#%%
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=0, stratify=Y)
#%%
print(f'Shape of train Dataset: {X_train.shape, y_train.shape}')
print(f'Shape of test Dataset: {X_test.shape, y_test.shape}')
#%%
vec_size = 100

model = Sequential()
model.add(Embedding(vocab_size, vec_size, input_length=max_length))

model.add(Conv1D(70, 2, activation='relu'))
model.add(MaxPooling1D(2))
model.add(Dropout(0.2))

model.add(Conv1D(140, 2, activation='relu'))
model.add(MaxPooling1D(2))
model.add(Dropout(0.2))

model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(16, activation='relu'))
model.add(GlobalMaxPooling1D())

model.add(Dense(3, activation='softmax'))
#%%
model.summary()
#%%
plot_model(model, show_shapes=True, show_layer_names=True, dpi=64)

#%%
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer= SGD(), loss='categorical_crossentropy', metrics=['accuracy'])
r = model.fit(X_train, y_train, epochs=50, batch_size=50, validation_data=(X_test, y_test))
# %%
import math   
print(f'The traning accuracy of model is {math.floor(r.history["accuracy"][-1] * 100)} %')
print(f'The validation accuracy of model is {math.floor(r.history["val_accuracy"][-1] * 100)} %')
#%%
print(f'The traning loss of model is {r.history["loss"][-1]:.4}')
print(f'The validation loss of model is {r.history["val_loss"][-1]:.4}')
#%%
plt.rcParams['figure.dpi'] = 75
plt.rcParams['figure.figsize'] = [20,7]
#%%
plt.title('Loss')
plt.plot(r.history['loss'], ".:", label='loss', linewidth=1.5, color="red")
plt.plot(r.history['val_loss'], ".:", label='val loss', linewidth=1.5, color="blue")
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()
#%%
plt.title('Accuracy')
plt.plot(r.history['accuracy'], ".:", label='accuracy', linewidth=1.5, color="red")
plt.plot(r.history['val_accuracy'], ".:", label='val accuracy', linewidth=1.5, color="blue")
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()
#%%
classes = ['Depression', 'Other', 'SuicideWatch']
# {Depression:0, Other:1 ,SuicideWatch:2}
#%%
text = input('Enter Your Message: ')
examples = text
# %%
text = get_clean(text)
en_text = encoded_text = token.texts_to_sequences(text)
en_text = pad_sequences(en_text, maxlen=max_length, padding='pre')
#%%
x = get_clean(text)
x = encoded_text = token.texts_to_sequences(x)
x = pad_sequences(x, maxlen=max_length, padding='pre')
y_pred = np.argmax(model.predict(x))
# %%
print(f'Sentence: {text}')
print(f'Result: Model has found symptoms of {classes[y_pred]} in the above sentence.')
# %%
