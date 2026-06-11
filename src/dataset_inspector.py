import os
from collections import Counter

# Struttura cartelle nel repository ActivityRecognition
ANNOTATIONS_DIR = os.path.join("..", "data", "InterX", "annotations")

FILE_ACTION_SETTING = os.path.join(ANNOTATIONS_DIR, "action_setting.txt")
FILE_FAMILIARITY = os.path.join(ANNOTATIONS_DIR, "familiarity.txt")
FILE_ALL = os.path.join(ANNOTATIONS_DIR, "all.txt")
FILE_TRAIN = os.path.join(ANNOTATIONS_DIR, "train.txt")
FILE_VAL = os.path.join(ANNOTATIONS_DIR, "val.txt")
FILE_TEST = os.path.join(ANNOTATIONS_DIR, "test.txt")

# Mappatura semantica dei livelli di familiarità (Interpersonal Stance)
FAMILIARITY_MAP = {
    "1": "Estranei / Sconosciuti (Strangers)",
    "2": "Casual Acquantances (Conoscenti)",
    "3": "Amici / Colleghi (Friends/Colleagues)",
    "4": "Relazione Intima / Stretta (Intimate)"
}

def read_lines(file_path):
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def inspect_interx():
    print("=" * 70)
    print("  INTER-X DATASET EXPLORATORY ANALYSIS (AGENT PERSPECTIVE)")
    print("=" * 70)

    # 1. Caricamento del catalogo delle azioni dell'agente
    actions = read_lines(FILE_ACTION_SETTING)
    print(f"\n[1] CATALOGO DELLE AZIONI PERCEPITE DALL'AGENTE:")
    print(f"  * Numero totale di azioni atomiche configurate: {len(actions)}")
    if actions:
        print(f"  * Esempi di azioni sociali: {actions[0]} (0), {actions[1]} (1), {actions[16]} (16)")

    # 2. Analisi degli Split (Training, Validation, Test)
    train_seqs = read_lines(FILE_TRAIN)
    val_seqs = read_lines(FILE_VAL)
    test_seqs = read_lines(FILE_TEST)
    all_seqs = read_lines(FILE_ALL)

    print(f"\n[2] DISTRIBUZIONE DELLE SEQUENZE NEGLI SPLIT:")
    print(f"  - Sequenze di Addestramento (Train): {len(train_seqs)}")
    print(f"  - Sequenze di Calibrazione (Val):    {len(val_seqs)}")
    print(f"  - Sequenze di Valutazione (Test):    {len(test_seqs)}")
    print(f"  - Totale sequenze indicizzate:        {len(all_seqs)}")

    # 3. Analisi della Familiarità Sociale (Modello di Relazione Multi-Agente)
    familiarity_labels = read_lines(FILE_FAMILIARITY)
    
    print(f"\n[3] ANALISI DELLE RELAZIONI INTERPERSONALI (FAMILIARITÀ):")
    if len(familiarity_labels) == len(all_seqs):
        fam_counts = Counter(familiarity_labels)
        for code, count in sorted(fam_counts.items()):
            label_name = FAMILIARITY_MAP.get(code, "Sconosciuto")
            percentage = (count / len(all_seqs)) * 100
            print(f"  - Livello {code} [{label_name}]: {count} interazioni ({percentage:.1f}%)")
    else:
        print("  ⚠️ Attenzione: Il file familiarity.txt non corrisponde 1:1 con all.txt")

    # 4. Decodifica Semantica di un esempio (Simulazione di Percezione dell'Agente)
    print(f"\n[4] SIMULAZIONE DECODIFICA AGENTE DI VISIONE (Esempio riga 5):")
    if len(all_seqs) > 5 and len(actions) > 0:
        sample_seq = all_seqs[5] # Scegliamo una riga di esempio
        sample_fam = familiarity_labels[5] if len(familiarity_labels) > 5 else "1"
        
        # Estraiamo i token strutturati GxxxTxxxAxxxRxxx
        # Esempio: G001T000A001R005
        try:
            group_id = sample_seq.split('T')[0] # G001
            task_id = 'T' + sample_seq.split('T')[1].split('A')[0] # T000
            action_idx = int(sample_seq.split('A')[1].split('R')[0]) # 1
            rep_id = 'R' + sample_seq.split('R')[1] # R005
            
            action_name = actions[action_idx] if action_idx < len(actions) else "Indice Azione Fuori Tracciato"
            
            print(f"  * Codice grezzo percepito dai sensori: {sample_seq}")
            print(f"  * Analisi Contesto dell'Agente Head:")
            print(f"    └─ Identificativo Coppia Agenti: {group_id}")
            print(f"    └─ Obiettivo/Task della scena:   {task_id}")
            print(f"    └─ Azione Fisica Identificata:   {action_name} (Codice: A{action_idx:03d})")
            print(f"    └─ Livello Relazionale Attivo:   {FAMILIARITY_MAP.get(sample_fam)}")
        except Exception as e:
            print(f"  Impossibile decodificare la stringa d'esempio: {e}")
            
    print("=" * 70)

if __name__ == "__main__":
    inspect_interx()