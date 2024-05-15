"""
Script to get precision of action recognition using top-k predictions
"""

import pandas as pd

# Cargar los datos desde los archivos CSV
predicciones = pd.read_csv('topk_actions.csv')
ground_truth = pd.read_csv('gt_labels.txt', sep=' ')

# Separar las predicciones en listas
predicciones[['pred1', 'pred2', 'pred3', 'pred4', 'pred5']] = predicciones['predictions'].str.split(';', expand=True)

# Función para calcular las precisiones Top-3 y Top-5
def calcular_precisiones(row):
    # Obtener las predicciones y las etiquetas ground truth
    predicciones = row[['pred1', 'pred2', 'pred3', 'pred4', 'pred5']].tolist()
    etiquetas = ground_truth[ground_truth['video_id'] == row['video_id']]['gt_actions'].values[0].split(';')

    # Calcular aciertos para Top-3 y Top-5
    aciertos_top3 = any(pred in etiquetas for pred in predicciones[:3])
    aciertos_top5 = any(pred in etiquetas for pred in predicciones[:5])

    return aciertos_top3, aciertos_top5

# Aplicar la función a cada fila y obtener las precisiones
precisiones = predicciones.apply(calcular_precisiones, axis=1)
predicciones['acierto_top3'], predicciones['acierto_top5'] = zip(*precisiones)

# Calcular la precisión global
precision_top3 = predicciones['acierto_top3'].mean()
precision_top5 = predicciones['acierto_top5'].mean()

print(f'Precisión Top-3: {precision_top3:.6f}')
print(f'Precisión Top-5: {precision_top5:.6f}')
