# Predicción de Lipofilicidad Molecular con GATv2 🧪💊

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Geometric](https://img.shields.io/badge/PyG-Graph_Neural_Networks-green.svg)](https://www.pyg.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Este repositorio contiene una implementación avanzada de **Redes de Neuronas de Grafos (GNN)** para predecir la lipofilicidad de compuestos químicos. El proyecto fue desarrollado para el Torneo GNN 2025-26, logrando superar el *baseline* mediante el uso de atención dinámica y estrategias de entrenamiento robustas.



## 🚀 Resumen del Proyecto

La lipofilicidad es una propiedad física clave que determina la afinidad de una molécula por disolverse en grasas y solventes no polares. Es un factor crítico en el diseño de fármacos para predecir la farmacocinética y la permeabilidad de membrana.

### Diferenciales de esta implementación:
* **Arquitectura GATv2:** A diferencia de las GAT estándar, GATv2 implementa **atención dinámica**, permitiendo que cada nodo tenga una jerarquía de importancia de vecinos única.
* **Pooling Híbrido:** Combinación de `global_mean_pool` y `global_max_pool` para una representación global más rica.
* **Control de Sobreajuste:** Integración de *Early Stopping* y *Learning Rate Scheduling*.

## 🧠 Arquitectura del Modelo

El modelo se basa en capas de atención dinámica que permiten capturar interacciones locales específicas entre átomos:

1.  **Mensaje y Update:** 3 capas `GATv2Conv` con 128 canales ocultos y **4 heads** de atención.
2.  **Agregación (Pooling):** Concatenación del promedio y el máximo de los embeddings de los nodos, resultando en un vector de 256 dimensiones.
3.  **Salida:** MLP de 3 capas con funciones de activación ReLU y **Dropout (0.3)** para la regresión final.



## 📊 Resultados

El modelo demostró una alta capacidad de generalización sobre el conjunto de test:

* **RMSE Final (Test):** `0.815391`
* **Baseline Referencia:** `0.8730`
* **Función de Pérdida:** Huber Loss (robusta ante *outliers*).

### Análisis Visual
El código genera automáticamente:
* **True vs Predicted Plot:** Para observar la dispersión y ajuste de la regresión.
* **Residual Heatmap:** Para analizar la densidad del error y posibles sesgos sistemáticos.



## 🛠️ Instalación y Uso

1.  **Clonar el repositorio:**
    ```bash
    git clone [https://github.com/tu-usuario/Prediccion-Lipofilicidad-Molecular.git](https://github.com/tu-usuario/Prediccion-Lipofilicidad-Molecular.git)
    cd Prediccion-Lipofilicidad-Molecular
    ```

2.  **Instalar dependencias:**
    ```bash
    pip install torch torch-geometric pandas matplotlib seaborn scikit-learn
    ```

3.  **Ejecutar:**
    El dataset se descarga automáticamente mediante la clase `TournamentDataset`. Solo tienes que ejecutar las celdas del notebook o el script de Python proporcionado.

## 📈 Pipeline de Entrenamiento

* **Optimización:** Adam con `weight_decay=5e-4`.
* **Scheduler:** `ReduceLROnPlateau` (factor 0.5, paciencia 10) para ajustar el LR dinámicamente.
* **Early Stopping:** Paciencia de 25 épocas con guardado del mejor modelo (`best_model.pt`).

## 📚 Bibliografía
* **Brody, S., Alon, U., & Yahav, E.** (2021). *How Attentive are Graph Attention Networks?* (GATv2).
* **Fey, M., & Lenssen, J. E.** (2019). *Fast Graph Representation Learning with PyTorch Geometric*.
* **MoleculeNet:** A Benchmark for Deep Learning on Molecular Design.

---
**Autor:** Yushetf López Jiménez  
**Proyecto:** GNN Tournament 2025-26
