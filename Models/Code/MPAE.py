import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import warnings

# Suppress TensorFlow information messages
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# ======== CONFIGURATION ========
TRAIN_CSV_PATH = .MPAEdataMPAEBullet2300.csv
MODEL_SAVE_PATH = .MPAEBullet2300mpae_model_2300_Bullet 
TOKENIZER_SAVE_PATH = .MPAEBullet2300move_tokenizer_2300_Bullet
MAX_MOVES = 75
# ===============================

class MPAE(Model)
    def __init__(self, input_dim, kwargs)
        super().__init__(kwargs)
        self.input_dim = input_dim
        self.threshold = tf.Variable(0.0, trainable=False)
        self.mean_error = tf.Variable(0.0, trainable=False)
        self.std_error = tf.Variable(1.0, trainable=False)
        
        self.encoder = tf.keras.Sequential([
            Dense(128, activation='relu', input_shape=(input_dim,)),
            Dense(64, activation='relu'),
            Dense(32, activation='relu')
        ])
        
        self.decoder = tf.keras.Sequential([
            Dense(64, activation='relu'),
            Dense(128, activation='relu'),
            Dense(input_dim, activation='sigmoid')
        ])

    def call(self, inputs)
        return self.decoder(self.encoder(inputs))

    def train_step(self, data)
        x, y = data
        with tf.GradientTape() as tape
            reconstructions = self(x, training=True)
            loss = tf.reduce_mean(tf.square(y - reconstructions))
        
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {'loss' loss}

class TokenizerWrapper(tf.Module)
    def __init__(self, tokenizer)
        super().__init__()
        self.tokenizer = tokenizer
        self.vocabulary_size = tf.Variable(
            tokenizer.vocabulary_size(), 
            trainable=False,
            dtype=tf.int64
        )
        
    @tf.function(input_signature=[tf.TensorSpec([None], dtype=tf.string)])
    def serve(self, inputs)
        return self.tokenizer(inputs)

def preprocess_data()
    # Load data with strict type enforcement
    df = pd.read_csv(TRAIN_CSV_PATH, dtype={'Move' str})
    
    # Clean data before processing
    df = df.dropna(subset=['Move'])
    df['Move'] = df['Move'].str.strip().replace('', np.nan).dropna()
    
    games = df.groupby('Game ID')['Move'].apply(list)
    games = games[games.apply(len)  0]  # Remove empty game sequences

    # Text vectorization with robust adaptation
    tokenizer = TextVectorization(
        max_tokens=1000,
        output_mode='int',
        standardize=None,
        output_sequence_length=MAX_MOVES
    )
    
    # Prepare clean text corpus
    text_corpus = games.explode()
    text_corpus = text_corpus[~text_corpus.isin(['', 'nan', 'NaN'])]
    tokenizer.adapt(text_corpus.to_list())

    # Convert moves to sequences with validation
    seq_int = []
    for seq in games
        validated_seq = [str(move).strip() for move in seq if move and str(move).strip()]
        if len(validated_seq) == 0
            continue
        seq_int.append(tokenizer([ .join(validated_seq)]).numpy()[0])
    
    padded = pad_sequences(seq_int, maxlen=MAX_MOVES, padding='post', truncating='post')
    padded = padded  tokenizer.vocabulary_size()

    # Process numerical features with NaN handling
    features = df.groupby('Game ID').agg({
        'Eval Diff' list,
        'Time Spent' list,
        'Board Complexity' list
    })

    num_games = len(games)
    combined_data = np.zeros((num_games, MAX_MOVES, 4), dtype=np.float32)

    for i, game_id in enumerate(games.index)
        for j, feat_name in enumerate(['Eval Diff', 'Time Spent', 'Board Complexity'])
            feat_values = features.loc[game_id, feat_name]
            feat_values = [x for x in feat_values if not np.isnan(x)]
            
            padded_feat = pad_sequences(
                [feat_values],
                maxlen=MAX_MOVES,
                padding='post',
                truncating='post',
                dtype=np.float32
            )[0]
            combined_data[i, , j+1] = padded_feat

        combined_data[i, , 0] = padded[i]

    # Normalize numerical features with mask
    mask = combined_data[..., 0] != 0
    for feat_idx in range(1, 4)
        feature_data = combined_data[..., feat_idx]
        valid_values = feature_data[mask]
        valid_values = valid_values[~np.isnan(valid_values)]
        
        feat_mean = np.nanmean(valid_values)
        feat_std = np.nanstd(valid_values) + 1e-8
        
        combined_data[..., feat_idx] = np.where(
            mask,
            (feature_data - feat_mean)  feat_std,
            0.0
        )

    return combined_data.reshape(num_games, -1), tokenizer

def train_model()
    data, tokenizer = preprocess_data()
    input_dim = data.shape[1]
    
    model = MPAE(input_dim)
    model.compile(optimizer='adam', loss='mse')
    model.fit(data, data, epochs=50, batch_size=32, validation_split=0.1)
    
    # Calculate dynamic threshold
    reconstructions = model.predict(data)
    errors = tf.reduce_mean(tf.square(data - reconstructions), axis=1)
    model.threshold.assign(tf.reduce_mean(errors) + 2tf.math.reduce_std(errors))
    
    # Save model with signature
    model.save(MODEL_SAVE_PATH, save_format=tf)
    
    # Save tokenizer with serving signature
    tokenizer_module = TokenizerWrapper(tokenizer)
    tf.saved_model.save(
        tokenizer_module,
        TOKENIZER_SAVE_PATH,
        signatures={
            'serving_default' tokenizer_module.serve.get_concrete_function()
        }
    )
    
    print(fModel saved to {MODEL_SAVE_PATH})
    print(fTokenizer saved to {TOKENIZER_SAVE_PATH})

if __name__ == __main__
    train_model()