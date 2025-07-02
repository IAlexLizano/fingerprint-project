# Configuraci�n del Sistema de Autenticaci�n de Huellas Dactilares

# Configuraci�n del modelo
MODEL_CONFIG = {
    "input_shape": (128, 128, 1),
    "embedding_dim": 128,
    "model_path": "models/saved/siamese_model.h5"
}

# Configuraci�n del dataset
DATASET_CONFIG = {
    "dataset_path": "data/dataset",
    "embeddings_path": "data/embeddings",
    "embeddings_file": "data/embeddings/user_embeddings.npy",
    "users_file": "data/embeddings/users.json"
}

# Configuraci�n de entrenamiento
TRAINING_CONFIG = {
    "default_epochs": 50,
    "default_batch_size": 32,
    "default_validation_split": 0.2,
    "learning_rate": 0.0001
}

# Configuraci�n de autenticaci�n
AUTH_CONFIG = {
    "default_threshold": 0.5,
    "max_images_per_user": 10
}

# Configuraci�n del servidor
SERVER_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": True
}
