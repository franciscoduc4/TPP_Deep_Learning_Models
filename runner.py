import polars as pl
# import d3rlpy
from typing import Optional
import os
# from processing.xml_processing import process_subject_data, create_mdp_dataset, get_all_subjects
from processing.xml_processing import process_subject_data, get_all_subjects
from custom.printer import print_debug, print_critical, print_error, print_warning, print_success, print_info

# Constantes
MODEL_DIR = "output/models"

def main():
    """
    Función principal para procesar datos, entrenar y evaluar modelos TD3+BC.

    Retorna:
    --------
    None
    """
    # Obtener todos los sujetos
    subject_ids = get_all_subjects()
    
    print_debug(f"Subject Ids encontrados: {subject_ids}")
    
    for subject_id in subject_ids:
        print(f"Procesando sujeto {subject_id}")
        
        # Procesar datos
        train_df, val_df, test_df = process_subject_data(subject_id)
        if train_df is None or val_df is None or test_df is None:
            print(f"No se pudieron procesar los datos para el sujeto {subject_id}")
            continue
        
        # # Convertir a MDPDataset
        # train_dataset = create_mdp_dataset(train_df)
        # val_dataset = create_mdp_dataset(val_df)
        # test_dataset = create_mdp_dataset(test_df)
        
        print_debug(f"{train_df}")
        print_debug(f"{val_df}")
        print_debug(f"{test_df}")
        
        # # Inicializar y entrenar el modelo
        # model = TD3BC()
        # model.train(train_dataset)
        
        # # Guardar el modelo
        # model_path = os.path.join(MODEL_DIR, f"td3bc_{subject_id}.d3")
        # model.save(model_path)
        
        # # Evaluar el modelo
        # val_policy_value = model.evaluate(val_dataset)
        # test_policy_value = model.evaluate(test_dataset)
        
        # print(f"Sujeto {subject_id} - Valor de política (validación): {val_policy_value:.2f}")
        # print(f"Sujeto {subject_id} - Valor de política (prueba): {test_policy_value:.2f}")

if __name__ == "__main__":
    main()