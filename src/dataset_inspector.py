import os
from collections import Counter

# Struttura cartelle nel repository ActivityRecognition
INTERX_DIR = os.path.join("..", "data", "InterX")
ANNOTATIONS_DIR = os.path.join(INTERX_DIR, "annotations")
TEXTS_DIR = os.path.join(INTERX_DIR, "texts")

FILE_ACTION_SETTING = os.path.join(ANNOTATIONS_DIR, "action_setting.txt")
FILE_FAMILIARITY = os.path.join(ANNOTATIONS_DIR, "familiarity.txt")
FILE_ALL = os.path.join(ANNOTATIONS_DIR, "all.txt")

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

def inspect_interx_correct():
    print("=" * 75)
    print("  INTER-X DATASET EXPLORATORY ANALYSIS (CORRECTED MAPPING)")
    print("=" * 75)

    actions = read_lines(FILE_ACTION_SETTING)
    all_seqs = read_lines(FILE_ALL)
    
    # Carichiamo la lista delle familiarità (una riga per ogni gruppo G)
    familiarity_lines = read_lines(FILE_FAMILIARITY)
    # Creiamo un dizionario di associazione: Gruppo -> Livello di familiarità
    # Es: { 1: 1, 2: 4, 3: 1 ... } dove la chiave è il numero del gruppo
    group_familiarity = {}
    for idx, line in enumerate(familiarity_lines, 1):
        group_familiarity[idx] = int(line)

    print(f"\n[1] STATISTICHE DI BASE:")
    print(f"  - Totale sequenze video indicizzate: {len(all_seqs)}")
    print(f"  - Numero totale di Gruppi di attori: {len(group_familiarity)}")
    print(f"  - Azioni fisiche a catalogo:         {len(actions)}")
    
    if os.path.exists(TEXTS_DIR):
        total_text_files = len([f for f in os.listdir(TEXTS_DIR) if f.endswith('.txt')])
        print(f"  - File di descrizione testuale trovati: {total_text_files}")

    # 2. Analisi corretta della Familiarità Sociale estesa a tutte le sequenze
    all_sequences_familiarity = []
    
    for seq in all_seqs:
        try:
            # Estraiamo il numero del gruppo eliminando la 'G' iniziale
            # Es: 'G059T001...' -> '059' -> int: 59
            group_num = int(seq.split('T')[0].replace('G', ''))
            # Recuperiamo la familiarità associata a quel gruppo
            fam_code = group_familiarity.get(group_num, 1)
            all_sequences_familiarity.append(fam_code)
        except:
            pass

    print(f"\n[2] DISTRIBUZIONE DELLA FAMILIARITÀ SU TUTTI I VIDEO:")
    fam_counts = Counter(all_sequences_familiarity)
    for code, count in sorted(fam_counts.items()):
        label_name = FAMILIARITY_MAP.get(code, "Sconosciuto")
        percentage = (count / len(all_seqs)) * 100
        print(f"  - Livello {code} [{label_name}]: {count} video ({percentage:.1f}%)")

    # 3. Simulazione Decodifica Multimodale ed Esplicabile di un esempio
    print(f"\n[3] SIMULAZIONE DECODIFICA AGENTE DI VISIONE (Riga 5):")
    if len(all_seqs) > 5:
        sample_seq = all_seqs[5] # G001T000A001R005
        
        try:
            group_id = sample_seq.split('T')[0] # G001
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
    inspect_interx_correct()