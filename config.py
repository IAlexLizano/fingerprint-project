# Configuración del Sistema de Autenticación de Huellas Dactilares

# Configuración del modelo
MODEL_CONFIG = {
    "input_shape": (128, 128, 1),
    "embedding_dim": 128,
    "model_path": "models/saved/siamese_model.h5"
}

# Configuración del dataset
DATASET_CONFIG = {
    "dataset_path": "data/dataset",
    "embeddings_path": "data/embeddings",
    "embeddings_file": "data/embeddings/user_embeddings.npy",
    "users_file": "data/embeddings/users.json"
}

# Configuración de entrenamiento
TRAINING_CONFIG = {
    "default_epochs": 50,
    "default_batch_size": 32,
    "default_validation_split": 0.2,
    "learning_rate": 0.0001
}

# Configuración de autenticación
AUTH_CONFIG = {
    "default_threshold": 0.5,
    "max_images_per_user": 10
}

# Configuración del servidor
SERVER_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": True
}
