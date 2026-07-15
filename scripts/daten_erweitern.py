# ============================================================
# Datensatz erweitern: alle Krebsfaelle + gleiche Menge gesunder Bilder
# in den BA-Ordner train_images/ kopieren.
# ============================================================
# Standard: TROCKENLAUF (zeigt nur, was fehlt, kopiert nichts).
# Zum echten Kopieren unten QUELLE auf deinen 300-GB-Download setzen.

import os
import shutil
import pandas as pd

# ---- EINSTELLUNGEN --------------------------------------------------
# Pfad zum vollstaendigen Kaggle-Download (Ordner, der die Patienten-Unterordner
# enthaelt, also .../train_images). Leer lassen = nur Trockenlauf.
QUELLE = r'C:\Users\edagu\Downloads\train_images'   # vollstaendiger Kaggle-Download
ZIEL   = 'train_images'          # dein BA-Ordner (relativ zum Projekt)
CSV    = 'train.csv'
N_GESUND_ZIEL = 1158             # gleiche Menge wie Krebsfaelle
SEED   = 42
# --------------------------------------------------------------------

df = pd.read_csv(CSV)


def ist_lokal(pid, iid):
    return os.path.exists(os.path.join(ZIEL, str(pid), f'{iid}.dcm'))


# 1) Zielauswahl bestimmen
krebs = df[df['cancer'] == 1].copy()                       # alle 1158 Krebsfaelle
gesund_all = df[df['cancer'] == 0].copy()

# Gesunde: vorhandene behalten, dann reproduzierbar auffuellen bis N_GESUND_ZIEL
gesund_all['lokal'] = gesund_all.apply(lambda z: ist_lokal(z['patient_id'], z['image_id']), axis=1)
gesund_lokal = gesund_all[gesund_all['lokal']]
gesund_rest  = gesund_all[~gesund_all['lokal']]
n_fehlt_gesund = max(0, N_GESUND_ZIEL - len(gesund_lokal))
gesund_extra = gesund_rest.sample(n=min(n_fehlt_gesund, len(gesund_rest)), random_state=SEED)
gesund_ziel = pd.concat([gesund_lokal, gesund_extra])

auswahl = pd.concat([krebs, gesund_ziel]).drop_duplicates(subset=['patient_id', 'image_id'])
print(f'Zielauswahl: {len(krebs)} Krebs + {len(gesund_ziel)} gesund = {len(auswahl)} Bilder')

# 2) Status jeder Datei bestimmen: schon da / muss kopiert werden / im Download fehlt
zeilen = []
schon_da = kopiert = quelle_fehlt = 0
trockenlauf = not (QUELLE and os.path.isdir(QUELLE))
if trockenlauf:
    print('\n*** TROCKENLAUF *** (QUELLE nicht gesetzt/gefunden -> es wird nichts kopiert)\n')

for _, z in auswahl.iterrows():
    pid, iid = str(z['patient_id']), z['image_id']
    ziel_datei = os.path.join(ZIEL, pid, f'{iid}.dcm')
    if os.path.exists(ziel_datei):
        status = 'schon_vorhanden'
        schon_da += 1
    else:
        quelle_datei = os.path.join(QUELLE, pid, f'{iid}.dcm') if QUELLE else ''
        if QUELLE and os.path.exists(quelle_datei):
            if not trockenlauf:
                os.makedirs(os.path.join(ZIEL, pid), exist_ok=True)
                shutil.copy2(quelle_datei, ziel_datei)
                status = 'kopiert'
                kopiert += 1
            else:
                status = 'wuerde_kopiert'
        else:
            status = 'fehlt_zu_kopieren' if trockenlauf else 'IM_DOWNLOAD_NICHT_GEFUNDEN'
            quelle_fehlt += 1
    zeilen.append({'patient_id': pid, 'image_id': iid, 'cancer': int(z['cancer']), 'status': status})

# 3) Auswahl + Status als CSV speichern (Liste, welche Patienten/Bilder gebraucht werden)
out = pd.DataFrame(zeilen)
out.to_csv('datenauswahl_erweitert.csv', index=False, encoding='utf-8')

# 4) Zusammenfassung
print('--- Zusammenfassung ---')
print(f'Schon im BA-Ordner:        {schon_da}')
if trockenlauf:
    print(f'Noch zu kopieren (fehlen): {quelle_fehlt}')
    print('\nSetze oben QUELLE auf deinen Download-Pfad und starte erneut, um sie zu kopieren.')
else:
    print(f'Neu kopiert:               {kopiert}')
    print(f'Im Download nicht gefunden:{quelle_fehlt}')
print(f'\nDetail-Liste gespeichert in: datenauswahl_erweitert.csv')
print('Benoetigte, aber noch fehlende Patienten stehen dort mit Status fehlt_zu_kopieren / IM_DOWNLOAD_NICHT_GEFUNDEN.')
