#!/usr/bin/env python3
"""
Documentación de endpoints de la API de huellas dactilares
"""

def show_endpoints_documentation():
    """Muestra documentación completa de todos los endpoints"""
    
    print("📚 DOCUMENTACIÓN DE ENDPOINTS - API HUELLAS DACTILARES")
    print("=" * 65)
    
    print("\n🏠 ENDPOINTS BÁSICOS")
    print("-" * 30)
    print("GET    /                    - Información básica de la API")
    print("GET    /health              - Estado de salud del sistema")
    print("GET    /docs               - Documentación interactiva (Swagger)")
    print("GET    /redoc              - Documentación alternativa (ReDoc)")
    
    print("\n👥 GESTIÓN DE USUARIOS")
    print("-" * 30)
    print("GET    /users              - Lista básica de usuarios")
    print("GET    /users/detailed     - Lista detallada con embeddings")
    print("GET    /users/{username}   - Información detallada de un usuario")
    print("POST   /register           - Registrar nuevo usuario")
    print("DELETE /users/{username}   - Eliminar usuario específico")
    print("DELETE /users              - ⚠️  Eliminar TODOS los usuarios")
    
    print("\n🧠 EMBEDDINGS Y DATOS")
    print("-" * 30)
    print("GET    /users/{username}/embeddings  - Embeddings de un usuario")
    print("GET    /dataset/info                 - Información del dataset")
    
    print("\n🔐 AUTENTICACIÓN")
    print("-" * 30)
    print("POST   /authenticate       - Autenticar usuario por huella")
    
    print("\n🤖 MODELO Y ENTRENAMIENTO")
    print("-" * 30)
    print("POST   /train              - Entrenar el modelo siamés")
    print("POST   /model/load         - Cargar modelo entrenado")
    
    print("\n📝 DETALLES DE ENDPOINTS PRINCIPALES")
    print("=" * 65)
    
    print("\n1️⃣ REGISTRO DE USUARIO")
    print("   POST /register")
    print("   Cuerpo: {")
    print('     "username": "string",')
    print('     "images": ["base64_image1", "base64_image2", ...]')
    print("   }")
    print("   Respuesta: información de registro y conteo de embeddings")
    
    print("\n2️⃣ AUTENTICACIÓN")
    print("   POST /authenticate")
    print("   Cuerpo: {")
    print('     "image": "base64_image"')
    print("   }")
    print("   Respuesta: resultado de autenticación y score de similitud")
    
    print("\n3️⃣ INFORMACIÓN DETALLADA DE USUARIO")
    print("   GET /users/{username}")
    print("   Respuesta incluye:")
    print("     - Información básica del usuario")
    print("     - Lista completa de embeddings")
    print("     - Estadísticas de embeddings (mean, std, min, max)")
    print("     - Rutas de imágenes")
    
    print("\n4️⃣ SOLO EMBEDDINGS")
    print("   GET /users/{username}/embeddings")
    print("   Respuesta incluye:")
    print("     - Embeddings del usuario")
    print("     - Dimensión de embeddings")
    print("     - Estadísticas detalladas")
    
    print("\n5️⃣ ELIMINACIÓN MASIVA ⚠️")
    print("   DELETE /users")
    print("   ⚠️  CUIDADO: Elimina TODOS los usuarios")
    print("   - Todos los usuarios registrados")
    print("   - Todos los embeddings")
    print("   - Todas las imágenes guardadas")
    print("   - Operación irreversible")
    
    print("\n6️⃣ ENTRENAMIENTO")
    print("   POST /train")
    print("   Cuerpo: {")
    print('     "dataset_path": "data/fvc2004_db1a_dataset.json",')
    print('     "epochs": 20,')
    print('     "batch_size": 64,')
    print('     "validation_split": 0.2')
    print("   }")
    
    print("\n🔧 EJEMPLOS DE USO CON CURL")
    print("=" * 65)
    
    print("\n# Obtener lista de usuarios")
    print("curl -X GET http://localhost:8000/users")
    
    print("\n# Obtener información detallada de un usuario")
    print("curl -X GET http://localhost:8000/users/juan_perez")
    
    print("\n# Obtener solo embeddings de un usuario")
    print("curl -X GET http://localhost:8000/users/juan_perez/embeddings")
    
    print("\n# Eliminar un usuario específico")
    print("curl -X DELETE http://localhost:8000/users/juan_perez")
    
    print("\n# ⚠️ Eliminar TODOS los usuarios (usar con cuidado)")
    print("curl -X DELETE http://localhost:8000/users")
    
    print("\n# Verificar salud del sistema")
    print("curl -X GET http://localhost:8000/health")
    
    print("\n📊 ESTRUCTURA DE RESPUESTAS")
    print("=" * 65)
    
    print("\n✅ Respuesta exitosa típica:")
    print("{")
    print('  "success": true,')
    print('  "message": "Operación exitosa",')
    print('  "data": { ... }')
    print("}")
    
    print("\n❌ Respuesta de error típica:")
    print("{")
    print('  "success": false,')
    print('  "message": "Descripción del error"')
    print("}")
    
    print("\n🔗 ACCESO A LA DOCUMENTACIÓN INTERACTIVA")
    print("=" * 65)
    print("Una vez que la API esté ejecutándose:")
    print("  - Swagger UI: http://localhost:8000/docs")
    print("  - ReDoc:      http://localhost:8000/redoc")
    
    print("\n🚀 INICIO RÁPIDO")
    print("=" * 65)
    print("1. Iniciar la API:     python main.py")
    print("2. Probar endpoints:   python test_endpoints.py")
    print("3. Ver docs:           http://localhost:8000/docs")

if __name__ == "__main__":
    show_endpoints_documentation()
