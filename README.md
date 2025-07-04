# Sistema de Autenticación de Huellas Dactilares

Un sistema backend completo en Python para autenticación de huellas dactilares usando redes neuronales siamesas. Entrenado con el dataset FVC2004 DB1_A para reconocimiento y validación de huellas dactilares.

## Características

- **Red Neuronal Siamesa**: Arquitectura especializada para comparación de huellas dactilares
- **Dataset FVC2004**: Entrenamiento con dataset estándar de huellas dactilares
- **API REST**: Interfaz completa con FastAPI para registro y autenticación
- **Autenticación 1:N**: Identificación de usuarios contra toda la base de datos
- **Procesamiento de Imágenes**: Preprocesamiento automático optimizado para huellas
- **Gestión de Embeddings**: Almacenamiento eficiente de características extraídas

## Estructura del Proyecto

```
project-fingerprint/
├── main.py                     # API REST principal
├── train_fvc_model.py         # Script de entrenamiento con FVC2004
├── requirements.txt           # Dependencias del proyecto
├── models/                    # Modelos de redes neuronales
│   └── siamese_network.py    # Red neuronal siamesa
├── services/                  # Servicios principales
│   └── fingerprint_service.py # Servicio de autenticación
├── api/                       # Esquemas de la API
│   └── schemas.py            # Modelos de datos para API
├── data/                      # Datos y gestión
│   ├── dataset_manager.py    # Gestor de datasets
│   ├── fvc2004_db1a_dataset.json  # Dataset FVC2004 DB1_A
│   └── DB1_A/                # Imágenes del dataset FVC2004
└── training/                  # Entrenamiento
    └── trainer.py            # Entrenador del modelo
├── services/              # Lógica de negocio
│   ├── __init__.py
│   └── fingerprint_service.py # Servicio principal
├── api/                   # Esquemas de API
│   ├── __init__.py
│   └── schemas.py         # Modelos Pydantic
├── utils/                 # Utilidades
│   ├── __init__.py
│   └── dataset_generator.py # Generador de datasets
└── examples/              # Ejemplos de uso
    ├── __init__.py
    └── example_usage.py   # Ejemplo completo
```

## Instalación y Uso

### 1. Clonar el repositorio

```bash
git clone <repository-url>
cd project-fingerprint
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Entrenar el modelo (opcional)

Si quieres reentrenar el modelo con el dataset FVC2004:

```bash
python train_fvc_model.py
```

### 4. Ejecutar la API REST

```bash
python main.py
```

El servidor estará disponible en `http://localhost:8000`

### 5. Documentación de la API

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## ⚠️ **Resultados del Entrenamiento y Soluciones**

### Problemas Identificados y Resueltos:

1. **🔧 Accuracy Estancada (50%)**:

   - **Problema**: Red no aprendía, accuracy fija en 50%
   - **Solución**: Arquitectura optimizada específica para huellas dactilares
   - **Archivo**: `models/fingerprint_siamese.py`

2. **🔧 Learning Rate**:

   - **Cambio**: De 0.0001 a 0.0005 para mejor convergencia
   - **Resultado**: Modelo comienza a aprender gradualmente

3. **🔧 Arquitectura Simplificada**:

   - **Antes**: Distancia euclidiana compleja
   - **Ahora**: Diferencia absoluta + capas densas
   - **Beneficio**: Más estable y fácil de entrenar

4. **🔧 Callbacks Automáticos**:
   - Early Stopping: Para en 5 épocas sin mejora
   - Reduce LR: Reduce learning rate automáticamente

### Uso del Nuevo Modelo:

```bash
# Entrenamiento mejorado
python train_fvc_model.py
```

**Archivos Clave**:

- `models/fingerprint_siamese.py` - Red optimizada
- `train_fvc_model.py` - Script de entrenamiento corregido

### Estructura del Proyecto Actualizada:

```
project-fingerprint/
├── main.py                        # API REST principal
├── train_fvc_model.py             # ✅ Script corregido
├── models/
│   ├── siamese_network.py         # Red original
│   └── fingerprint_siamese.py     # ✅ Red optimizada
├── services/
│   └── fingerprint_service.py     # Servicio de autenticación
└── data/
    ├── fvc2004_db1a_dataset.json  # Dataset FVC2004 DB1_A
    └── DB1_A/                     # Imágenes (800 muestras, 100 usuarios)
```

## API Endpoints

### Autenticación y Registro

#### POST `/register`

Registra un nuevo usuario con sus huellas dactilares.

**Request Body**:

```json
{
  "username": "usuario123",
  "images": ["base64_image_1", "base64_image_2", "base64_image_3"]
}
```

**Response**:

```json
{
  "success": true,
  "message": "Usuario 'usuario123' registrado exitosamente con 3 embeddings.",
  "username": "usuario123",
  "embedding_count": 3
}
```

#### POST `/authenticate`

Autentica un usuario basado en su huella dactilar.

**Request Body**:

```json
{
  "image": "base64_image"
}
```

**Response**:

```json
{
  "success": true,
  "authenticated": true,
  "username": "usuario123",
  "similarity_score": 0.85,
  "message": "Usuario autenticado: usuario123"
}
```

### Entrenamiento

#### POST `/train`

Entrena el modelo siamesa con un dataset.

**Request Body**:

```json
{
  "dataset_path": "data/datasets/training_dataset.json",
  "epochs": 50,
  "batch_size": 32,
  "validation_split": 0.2
}
```

### Gestión de Datos

#### GET `/dataset/info`

Obtiene información del dataset actual.

#### GET `/users`

Lista todos los usuarios registrados.

#### DELETE `/users/{username}`

Elimina un usuario del sistema.

### Utilidades

#### GET `/health`

Verifica el estado de salud del sistema.

#### POST `/model/load`

Carga el modelo entrenado.

## Formato del Dataset

El sistema espera un archivo JSON con el siguiente formato:

```json
[
  {
    "username": "usuario1",
    "image_paths": [
      "/path/to/image1.jpg",
      "/path/to/image2.jpg",
      "/path/to/image3.jpg"
    ]
  },
  {
    "username": "usuario2",
    "image_paths": ["/path/to/image4.jpg", "/path/to/image5.jpg"]
  }
]
```

## Generación de Datasets

El sistema incluye un generador de datasets sintéticos para pruebas:

```python
from utils.dataset_generator import DatasetGenerator

# Crear generador
generator = DatasetGenerator()

# Generar dataset de muestra
dataset_path = generator.create_sample_dataset()

# Generar dataset completo
training_dataset = generator.generate_training_dataset(
    num_users=20,
    images_per_user=5,
    output_filename="training_dataset.json"
)
```

## Arquitectura de la Red Neuronal

### Red Siamesa

- **Entrada**: Dos imágenes de 128x128 píxeles en escala de grises
- **Arquitectura**: 4 capas convolucionales + capas densas
- **Salida**: Score de similitud entre 0 y 1

### Preprocesamiento

1. Conversión a escala de grises
2. Redimensionado a 128x128 píxeles
3. Normalización a [0, 1]
4. Aplicación de transformaciones de data augmentation

### Embeddings

- Dimensión: 128 características
- Almacenamiento: Archivo .npy
- Comparación: Distancia euclidiana

## Configuración

### Parámetros del Modelo

- **Tamaño de entrada**: 128x128x1 (configurable)
- **Dimensión de embedding**: 128 (configurable)
- **Umbral de similitud**: 0.5 (configurable)

### Parámetros de Entrenamiento

- **Épocas**: 50 (recomendado)
- **Batch size**: 32
- **Learning rate**: 0.0001
- **Validación**: 20% del dataset

## Ejemplos de Uso

### Registro de Usuario

```python
import requests
import base64

# Cargar imagen
with open("fingerprint.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

# Registrar usuario
response = requests.post("http://localhost:8000/register", json={
    "username": "john_doe",
    "images": [image_data]
})

print(response.json())
```

### Autenticación

```python
# Autenticar usuario
response = requests.post("http://localhost:8000/authenticate", json={
    "image": image_data
})

result = response.json()
if result["authenticated"]:
    print(f"Usuario autenticado: {result['username']}")
else:
    print("Autenticación fallida")
```

## Dependencias Principales

- **FastAPI**: Framework web
- **TensorFlow**: Redes neuronales
- **OpenCV**: Procesamiento de imágenes
- **NumPy**: Computación numérica
- **Pillow**: Manipulación de imágenes
- **scikit-learn**: Métricas de evaluación

## Contribución

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## Soporte

Para soporte técnico o preguntas, por favor abrir un issue en el repositorio.
