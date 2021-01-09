import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Lading the cleaned data csv file
path = "../input/cleaned-data-for-nlp-news-classification/cleaned_data.csv"
data = pd.read_csv(path)

# Tokenizing
vocab_size = 10000
embedding_dim = 32
max_length = 150
trunc_type = 'post'
oov_tok = '<OOV>'
padding_post = 'post'

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)
X = pad_sequences(X, padding=padding_post, maxlen=max_length, truncating=trunc_type)

# Building Model
keras.backend.clear_session()

model = tf.keras.Sequential([
    keras.layers.Embedding(vocab_size, embedding_dim, input_length=X.shape[1], input_shape=[None]),
    keras.layers.Bidirectional(keras.layers.LSTM(256, dropout=0.5, return_sequences=True)),
    keras.layers.Bidirectional(keras.layers.LSTM(128, dropout=0.5)),
    keras.layers.Dense(12, activation='softmax')
])

Adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

model.compile(loss='categorical_crossentropy', optimizer=Adam, metrics=['acc'])

# One Hot encoding the categories
Y = pd.get_dummies(data['category']).values

# Splitting data into train,valid,test¶
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_valid, X_test, Y_valid, Y_test = train_test_split(X_test, Y_test, test_size=0.5, random_state=42)


# Learning Rate Scheduler
def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1 ** (epoch / s)

    return exponential_decay_fn


exponential_decay_fn = exponential_decay(lr0=0.001, s=5)

lr_scheduler_ed = keras.callbacks.LearningRateScheduler(exponential_decay_fn)

# Early stopping
early_stopping_m = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

# Fitting Model
batch_size = 32
epoch = 15
history = model.fit(X_train, Y_train, epochs=epoch, batch_size=batch_size, validation_data=(X_valid, Y_valid),
                    callbacks=[lr_scheduler_ed, early_stopping_m], verbose=2)

# Model Predict
Y_pred = model.predict(X_test)
Y_pred = np.argmax(Y_pred, axis=1)

# Classification Report
Y_test_ = np.argmax(Y_test, axis=1)
print(classification_report(Y_test_, Y_pred))

# Confusion Matrix
cf_matrix = confusion_matrix(Y_test_, Y_pred)
Labels = ['ARTS , CULTURE & ENVIRONMENT', 'BLACK LIVES MATTER', 'BUSINESS & MONEY', 'ENTERTAINMENT', 'FOOD & TRAVEL',
          'GOOD NEWS', 'LAW & CRIME', 'PARENTING', 'POLITICS & RELIGION', 'SPORTS & EDUCATION', 'STYLE & BEAUTY',
          'WORLD NEWS']
plt.figure(figsize=(8, 8))
heatmap = sns.heatmap(cf_matrix, xticklabels=Labels, yticklabels=Labels, annot=True, fmt='d', color='blue')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('Confusion Matrix')
plt.show()


# Graph for Accuracy And Loss
def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.grid()
    plt.show()


plot_graphs(history, "acc")
plot_graphs(history, "loss")

# Graph for Learning rate ~ Exponential Decay
plt.plot(history.epoch, history.history["lr"], "o-")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title(" exponential_decay", fontsize=14)
plt.grid(True)
plt.show()
