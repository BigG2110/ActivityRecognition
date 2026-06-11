import os
from collections import Counter

# Struttura cartelle nel repository ActivityRecognition
INTERX_DIR = os.path.join("..", "data", "InterX")
ANNOTATIONS_DIR = os.path.join(INTERX_DIR, "annotations")
TEXTS_DIR = os.path.join(INTERX_DIR, "texts")

FILE_ACTION_SETTING = os.path.join(ANNOTATIONS_DIR, "action_setting.txt")
FILE_FAMILIARITY = os.path.join(ANNOTATIONS_DIR, "familiarity.txt")
FILE_ALL = os.path.join(ANNOTATIONS_DIR, "all.txt")
FILE_TRAIN = os.path.join(ANNOTATIONS_DIR, "train.txt")
FILE_VAL = os.path.join(ANNOTATIONS_DIR, "val.txt")
FILE_TEST = os.path.join(ANNOTATIONS_DIR, "test.txt")

# Mappatura semantica dei livelli di familiarità (Interpersonal Stance)
FAMILIARITY_MAP = {
    1: "Estranei / Sconosciuti (Strangers)",
    2: "Casual Acquaintances (Conoscenti)",
    3: "Amici / Colleghi (Friends/Colleagues)",
    4: "Relazione Intima / Stretta (Intimate)"
}

def read_lines(file_path):
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def inspect_interx_complete():
    print("=" * 75)
    print("  INTER-X DATASET EXPLORATORY ANALYSIS (COMPLETE REPORT)")
    print("=" * 75)

    # Caricamento file di testo
    actions = read_lines(FILE_ACTION_SETTING)
    all_seqs = read_lines(FILE_ALL)
    train_seqs = read_lines(FILE_TRAIN)
    val_seqs = read_lines(FILE_VAL)
    test_seqs = read_lines(FILE_TEST)
    familiarity_lines = read_lines(FILE_FAMILIARITY)
    
    # Dizionario di associazione: Gruppo -> Livello di familiarità
    group_familiarity = {}
    for idx, line in enumerate(familiarity_lines, 1):
        group_familiarity[idx] = int(line)

    # [1] STATISTICHE DI BASE E CATALOGO AZIONI
    print(f"\n[1] STATISTICHE DI BASE E CATALOGO AZIONI:")
    print(f"  - Totale sequenze video indicizzate: {len(all_seqs)}")
    print(f"  - Numero totale di Gruppi di attori: {len(group_familiarity)}")
    print(f"  - Azioni fisiche a catalogo:         {len(actions)}")
    if len(actions) > 16:
        print(f"  * Esempi di azioni sociali: {actions[0]} (0), {actions[1]} (1), {actions[16]} (16)")
    
    if os.path.exists(TEXTS_DIR):
        total_text_files = len([f for f in os.listdir(TEXTS_DIR) if f.endswith('.txt')])
        print(f"  - File di descrizione testuale trovati: {total_text_files}")

    # [2] DISTRIBUZIONE DELLE SEQUENZE NEGLI SPLIT
    total_splits = len(train_seqs) + len(val_seqs) + len(test_seqs)
    print(f"\n[2] DISTRIBUZIONE DELLE SEQUENZE NEGLI SPLIT:")
    if total_splits > 0:
        print(f"  - Sequenze di Training (Train):      {len(train_seqs)} ({len(train_seqs)/total_splits*100:.1f}%)")
        print(f"  - Sequenze di Calibrazione (Val):    {len(val_seqs)} ({len(val_seqs)/total_splits*100:.1f}%)")
        print(f"  - Sequenze di Valutazione (Test):    {len(test_seqs)} ({len(test_seqs)/total_splits*100:.1f}%)")
    else:
        print("  ⚠️ Errore nel calcolo degli split. Controlla train.txt, val.txt e test.txt")

    # [3] DISTRIBUZIONE DELLA FAMILIARITÀ SU TUTTI I VIDEO
    all_sequences_familiarity = []
    for seq in all_seqs:
        try:
            group_num = int(seq.split('T')[0].replace('G', ''))
            fam_code = group_familiarity.get(group_num, 1)
            all_sequences_familiarity.append(fam_code)
        except:
            pass

    print(f"\n[3] DISTRIBUZIONE DELLA FAMILIARITÀ SU TUTTI I VIDEO:")
    if all_sequences_familiarity:
        fam_counts = Counter(all_sequences_familiarity)
        for code, count in sorted(fam_counts.items()):
            label_name = FAMILIARITY_MAP.get(code, "Sconosciuto")
            percentage = (count / len(all_seqs)) * 100
            print(f"  - Livello {code} [{label_name}]: {count} video ({percentage:.1f}%)")

    # [4] SIMULAZIONE DECODIFICA AGENTE DI VISIONE (RIGA 5)
    print(f"\n[4] SIMULAZIONE DECODIFICA AGENTE DI VISIONE (Riga 5):")
    if len(all_seqs) > 5:
        sample_seq = all_seqs[5] # G001T000A001R005
        
        try:
            group_id = sample_seq.split('T')[0]
            group_num = int(group_id.replace('G', ''))
            action_idx = int(sample_seq.split('A')[1].split('R')[0])
            
            action_name = actions[action_idx] if action_idx < len(actions) else "Unknown Action"
            fam_code = group_familiarity.get(group_num, 1)
            
            print(f"  * Codice della scena catturato: {sample_seq}")
            print(f"  * Analisi Contesto (Head Agent):")
            print(f"    └─ Gruppo Attori Rilevato: {group_id}")
            print(f"    └─ Azione Fisica Decodificata:  {action_name} (A{action_idx:03d})")
            print(f"    └─ Livello Relazionale del Gruppo: {FAMILIARITY_MAP.get(fam_code)}")
            
            # Lettura della descrizione testuale associata
            text_file_path = os.path.join(TEXTS_DIR, f"{sample_seq}.txt")
            print(f"  * Spiegazione in Linguaggio Naturale (Mouth Agent):")
            if os.path.exists(text_file_path):
                descriptions = read_lines(text_file_path)
                print(f"    └─ Descrizione: \"{descriptions[0]}\"")
            else:
                print(f"    ⚠️ File {sample_seq}.txt non trovato in {TEXTS_DIR}")
        except Exception as e:
            print(f"  Errore decodifica esempio: {e}")
            
    print("=" * 75)

if __name__ == "__main__":
    inspect_interx_complete()