import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks, constraints
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

# ====== HARDCODED PATHS ======
TRAIN_DATA_PATH = './GSAE/data/GSAEBlitz2500.csv'
MODEL_PATH = './GSAE/Blitz2500/GSAE_model_2500Blitz'
PREPROCESSOR_PATH = './GSAE/Blitz2500/GSAE_preprocessor_2500Blitz.pkl'

# ====== IMPROVED MODEL CONFIG ======
CONFIG = {
    "latent_dim": 6,
    "encoder_units": [32, 16],
    "decoder_units": [16, 32],
    "epochs": 100,
    "batch_size": 256,
    "learning_rate": 1e-4,
    "validation_split": 0.2,
    "threshold_percentile": 99.5,
    "max_decimal": 4,
    "gradient_clip": 1.0,
    "regularization": 0.01,
    "random_state": 42
}

def clean_data(df):
    """Enhanced data cleaning with validation checks"""
    processed = df.copy()
    
    # Remove non-feature columns
    processed = processed.drop(columns=['Game ID', 'Game Format'], errors='ignore')
    
    # Convert to numeric and handle infinities
    processed = processed.apply(pd.to_numeric, errors='coerce')
    processed = processed.replace([np.inf, -np.inf], np.nan)
    
    # Drop rows with >2 missing features
    processed = processed.dropna(thresh=len(processed.columns)-2)
    
    # Remove near-constant columns
    col_std = processed.std()
    valid_cols = col_std[col_std > 0.1].index
    processed = processed[valid_cols]
    
    # Final NaN check
    processed = processed.dropna()
    
    # Dynamic value clipping
    quantiles = processed.quantile([0.01, 0.99])
    processed = processed.clip(
        lower=quantiles.loc[0.01], 
        upper=quantiles.loc[0.99], 
        axis=1
    )
    
    return processed

def build_autoencoder(input_dim):
    """Numerically stable autoencoder architecture"""
    inputs = layers.Input(shape=(input_dim,))
    
    # Encoder with regularization
    x = layers.Dense(
        CONFIG['encoder_units'][0], 
        activation='relu',
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.l2(CONFIG['regularization'])
    )(inputs)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(
        CONFIG['encoder_units'][1],
        activation='relu',
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.l2(CONFIG['regularization'])
    )(x)
    encoded = layers.Dense(
        CONFIG['latent_dim'], 
        activation='linear',  # Simpler activation for stability
        kernel_constraint=constraints.MaxNorm(3)
    )(x)
    
    # Decoder with regularization
    x = layers.Dense(
        CONFIG['decoder_units'][0],
        activation='relu',
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.l2(CONFIG['regularization'])
    )(encoded)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(
        CONFIG['decoder_units'][1],
        activation='relu',
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.l2(CONFIG['regularization'])
    )(x)
    decoded = layers.Dense(
        input_dim, 
        activation='linear',
        kernel_constraint=constraints.MaxNorm(3)
    )(x)
    
    return Model(inputs, decoded)

def main():
    # Ensure directories
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(PREPROCESSOR_PATH), exist_ok=True)

    # Load and clean data
    raw_data = pd.read_csv(TRAIN_DATA_PATH)
    processed = clean_data(raw_data)
    
    if processed.empty:
        raise ValueError("No valid data remaining after cleaning!")
    
    # Data validation
    print("\n=== Cleaned Data Summary ===")
    print("Remaining samples:", len(processed))
    print("Features:", list(processed.columns))
    print("NaN check:", processed.isna().sum().sum())
    print("Inf check:", np.isinf(processed.values).sum())
    
    # Scale data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(processed)
    
    # Scaling validation
    print("\n=== Scaled Data Validation ===")
    print("NaNs:", np.isnan(scaled_data).sum())
    print("Infs:", np.isinf(scaled_data).sum())
    print("Value range: [{:.2f}, {:.2f}]".format(
        scaled_data.min(), scaled_data.max()
    ))
    
    # Build model
    autoencoder = build_autoencoder(scaled_data.shape[1])
    
    # Custom optimizer with gradient clipping
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=CONFIG['learning_rate'],
        clipvalue=CONFIG['gradient_clip']
    )
    
    autoencoder.compile(
        optimizer=optimizer,
        loss='mae',
        metrics=['mae']
    )
    
    # Train model
    X_train, X_val = train_test_split(
        scaled_data,
        test_size=CONFIG['validation_split'],
        random_state=CONFIG['random_state']
    )
    
    history = autoencoder.fit(
        X_train, X_train,
        epochs=CONFIG['epochs'],
        batch_size=CONFIG['batch_size'],
        validation_data=(X_val, X_val),
        callbacks=[
            callbacks.EarlyStopping(
                patience=10, 
                restore_best_weights=True,
                monitor='val_mae'
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_mae',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ],
        verbose=1
    )
    
    # Calculate threshold
    val_reconstructions = autoencoder.predict(X_val, verbose=0)
    val_errors = np.mean(np.abs(val_reconstructions - X_val), axis=1)
    threshold = np.percentile(val_errors, CONFIG['threshold_percentile'])
    
    # Add epsilon to prevent NaN
    threshold += 1e-8
    
    # Save artifacts
    autoencoder.save(MODEL_PATH)
    with open(PREPROCESSOR_PATH, 'wb') as f:
        pickle.dump({
            'scaler': scaler,
            'threshold': threshold,
            'features': processed.columns.tolist(),
            'config': CONFIG
        }, f)
    
    print("\n=== Training Summary ===")
    print(f"Final Threshold: {threshold:.6f}")
    print(f"Model saved to {MODEL_PATH}")
    print(f"Preprocessor saved to {PREPROCESSOR_PATH}")

if __name__ == "__main__":
    main()