import json
import os
from collections import Counter

# Configurazione dei percorsi (assumendo la struttura della cartella data/)
DATA_DIR = os.path.join("..", "data", "PISC", "annotations")

FILE_TRAIN = os.path.join(DATA_DIR, "relation_trainidx.json")
FILE_VAL = os.path.join(DATA_DIR, "relation_validx.json")
FILE_TEST = os.path.join(DATA_DIR, "relation_testidx.json")
FILE_RELATIONSHIPS = os.path.join(DATA_DIR, "relationship.json")
FILE_OCCUPATIONS = os.path.join(DATA_DIR, "occupation.json")
FILE_INFO = os.path.join(DATA_DIR, "annotation_image_info.json")

# Mappatura standard delle classi del dataset PISC (Fine-grained)
PISC_CLASSES = {
    1: "Friends (Amici)",
    2: "Family (Famiglia)",
    3: "Couples (Coppie)",
    4: "Professional (Colleghi/Lavoro)",
    5: "Commercial (Venditore/Cliente)",
    6: "No Relation (Nessuna Relazione)"
}

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_dataset():
    print("="*50)
    print("ANALISI ESPLORATIVA DEL DATASET PISC")
    print("="*50)
    
    # 1. Analisi della suddivisione dei Dati (Splits)
    train_ids = load_json(FILE_TRAIN)
    val_ids = load_json(FILE_VAL)
    test_ids = load_json(FILE_TEST)
    
    total_images = len(train_ids) + len(val_ids) + len(test_ids)
    print("\n[1] SUDDIVISIONE DEI DATI (SPLITS):")
    print(f"  - Immagini di Training:   {len(train_ids)} ({len(train_ids)/total_images*100:.1f}%)")
    print(f"  - Immagini di Validation: {len(val_ids)} ({len(val_ids)/total_images*100:.1f}%)")
    print(f"  - Immagini di Test:       {len(test_ids)} ({len(test_ids)/total_images*100:.1f}%)")
    print(f"  - Totale Immagini:        {total_images}")

    # 2. Analisi delle Relazioni Sociali (Classi/Labels)
    relationships = load_json(FILE_RELATIONSHIPS)
    all_labels = []
    total_pairs = 0
    
    for img_id, pairs in relationships.items():
        for pair_key, label in pairs.items():
            all_labels.append(label)
            total_pairs += 1
            
    label_counts = Counter(all_labels)
    
    print("\n[2] DISTRIBUZIONE DELLE RELAZIONI SOCIALI (COPPIE):")
    print(f"  - Numero totale di coppie annotate: {total_pairs}")
    for label_code, count in sorted(label_counts.items()):
        class_name = PISC_CLASSES.get(label_code, "Sconosciuta")
        percentage = (count / total_pairs) * 100
        print(f"    Classe {label_code} -> {class_name}: {count} coppie ({percentage:.1f}%)")

    # 3. Analisi delle Occupazioni / Ruoli nel contesto
    occupations_data = load_json(FILE_OCCUPATIONS)
    all_occupations = []
    for item in occupations_data:
        for person_id, occ in item["occupation"].items():
            all_occupations.append(occ)
            
    occ_counts = Counter(all_occupations)
    print("\n[3] TOP 5 OCCUPAZIONI PIÙ FREQUENTI:")
    for occ, count in occ_counts.most_common(5):
        print(f"    - {occ}: {count} persone")

    # 4. Analisi delle Bounding Boxes (Densità del dataset)
    img_info = load_json(FILE_INFO)
    total_bboxes = sum(len(img.get("bbox", [])) for img in img_info)
    avg_bboxes = total_bboxes / len(img_info)
    
    print("\n[4] GEOMETRIA E SENSING (BOUNDING BOXES):")
    print(f"  - Numero totale di persone rilevate (bboxes): {total_bboxes}")
    print(f"  - Numero medio di persone per immagine: {avg_bboxes:.2f}")
    print("="*50)

if __name__ == "__main__":
    # Verifica che i file siano posizionati correttamente prima di avviare
    if not os.path.exists(FILE_TRAIN):
        print(f"Errore: Non trovo i file in {DATA_DIR}. Controlla la struttura delle cartelle!")
    else:
        analyze_dataset()
