import os
import pandas as pd
import numpy as np

# Configurazione dei percorsi nel repository ActivityRecognition
# Lo script si trova in src/, quindi sale di un livello per andare in data/csv
CSV_DIR = os.path.join("..", "data", "csv")

def inspect_activity_dataset():
    print("=" * 65)
    print("  DETAILED INSPECTION: HUMAN INTERACTION RECOGNITION DATASET")
    print("=" * 65)
    
    if not os.path.exists(CSV_DIR):
        print(f"Errore: La cartella {CSV_DIR} non esiste.")
        print("Verifica di aver posizionato i CSV in ActivityRecognition/data/csv/")
        return

    # Individuiamo le sotto-cartelle (le classi/azioni fisiche)
    subfolders = [f for f in os.listdir(CSV_DIR) if os.path.isdir(os.path.join(CSV_DIR, f))]
    
    if not subfolders:
        print(f"Nessuna cartella trovata in {CSV_DIR}")
        return

    print(f"\n[1] AZIONI FISICHE/INTERAZIONI TROVATE ({len(subfolders)} classi):")
    
    total_sequences = 0
    all_lengths = []
    class_stats = {}

    # Scansione di ogni cartella di interazione
    for class_name in subfolders:
        class_path = os.path.join(CSV_DIR, class_name)
        csv_files = [f for f in os.listdir(class_path) if f.endswith('.csv')]
        
        num_files = len(csv_files)
        total_sequences += num_files
        
        lengths_in_class = []
        
        # Analizziamo la durata (numero di frame) di ogni file in questa classe
        for csv_file in csv_files:
            file_path = os.path.join(class_path, csv_file)
            # index_col=0 serve perché la prima colonna è l'indice del frame (0, 1, 2...)
            df = pd.read_csv(file_path, index_col=0)
            lengths_in_class.append(len(df))
            all_lengths.append(len(df))
            
        if num_files > 0:
            avg_len = np.mean(lengths_in_class)
            min_len = np.min(lengths_in_class)
            max_len = np.max(lengths_in_class)
            class_stats[class_name] = {
                "count": num_files,
                "avg_frames": avg_len,
                "min_frames": min_len,
                "max_frames": max_len
            }
            print(f"  - '{class_name}': {num_files} video/sequenze")
            print(f"    └─ Durata frame: media {avg_len:.1f} (min: {min_len}, max: {max_len})")
        else:
            print(f"  - '{class_name}': 0 file CSV trovati.")

    print("\n" + "-"*50)
    print("[2] STATISTICHE GLOBALI DEL DATASET:")
    print(f"  - Numero totale di sequenze (esempi di addestramento): {total_sequences}")
    if all_lengths:
        print(f"  - Durata media di un'interazione: {np.mean(all_lengths):.1f} frame")
        print(f"  - Sequenza più corta: {np.min(all_lengths)} frame")
        print(f"  - Sequenza più lunga: {np.max(all_lengths)} frame")

    # 3. Ispezione profonda di una riga di esempio per la relazione tecnica
    print("\n" + "-"*50)
    print("[3] ESTRATTO STRUTTURA DATI PER RELAZIONE TECNICA:")
    
    # Prendiamo il primo file disponibile per mostrare la struttura
    sample_class = subfolders[0]
    sample_class_path = os.path.join(CSV_DIR, sample_class)
    sample_files = [f for f in os.listdir(sample_class_path) if f.endswith('.csv')]
    
    if sample_files:
        sample_path = os.path.join(sample_class_path, sample_files[0])
        df_sample = pd.read_csv(sample_path, index_col=0)
        
        print(f"  - File analizzato: {sample_class}/{sample_files[0]}")
        print(f"  - Numero totale di colonne nel file: {df_sample.shape[1]}")
        print(f"  - Interpretazione Multi-Agente (Sensing):")
        print(f"    * Colonne 0-33  (34 feature): Coordinate X,Y dei 17 keypoint di Persona 1")
        print(f"    * Colonne 34-67 (34 feature): Coordinate X,Y dei 17 keypoint di Persona 2")
        print("\n  - Anteprima coordinate spaziali dei primi 2 frame:")
        # Mostriamo solo le prime colonne per leggibilità
        print(df_sample.iloc[:2, :6]) 
        print("    ... [i dati continuano per tutte le 68 colonne di feature] ...")
    
    print("=" * 65)

if __name__ == "__main__":
    inspect_activity_dataset()