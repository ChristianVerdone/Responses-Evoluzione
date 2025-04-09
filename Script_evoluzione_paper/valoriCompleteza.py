import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carica il file Excel
file_path = r"C:\Users\marco\Desktop\RisposteEvoluzione.xlsx"  # Sostituisci con il nome del file
df = pd.read_excel(file_path)

# Visualizza un'anteprima dei dati
print(df.head())

# Pulisci e normalizza i dati (se necessario)
df['Completezza'] = pd.to_numeric(df['Completezza'], errors='coerce')

# Enumerare le risposte da 1 a 10
df['Risposta'] = df.groupby('LLM').cumcount() + 1

# Filtrare i dati per ottenere solo la completezza
df_completezza = df[['LLM', 'Risposta', 'Completezza']]

# Creare un istogramma per ogni valore di Risposta e salvarlo come PNG
num_risposte = df['Risposta'].max()

# Utilizzare una palette di colori diversi
palette = sns.color_palette("husl", len(df['LLM'].unique()))

for i in range(1, num_risposte + 1):
    plt.figure(figsize=(10, 6))
    sns.barplot(x='LLM', y='Completezza', data=df_completezza[df_completezza['Risposta'] == i], palette=palette)
    plt.title(f"Completezza per Risposta {i}")
    plt.ylabel("Completezza")
    plt.xlabel("LLM")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"completezza_risposta_{i}.png")
    plt.close()