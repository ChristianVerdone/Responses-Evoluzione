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

# Calcolare il valore medio della completezza per ciascun LLM
df_mean_completezza = df.groupby('LLM')['Completezza'].mean().reset_index()

# Creare un istogramma unico per tutti gli LLM con colori diversi
plt.figure(figsize=(15, 8))
palette = sns.color_palette("husl", len(df_mean_completezza))  # Utilizzare una palette di colori diversi
sns.barplot(x='LLM', y='Completezza', data=df_mean_completezza, palette=palette)
plt.title("Valore medio della Completezza per ciascun LLM", fontsize=16)
plt.ylabel("Completezza")
plt.xlabel("LLM")
plt.xticks(rotation=45)
plt.tight_layout()

# Salva o mostra il grafico
plt.savefig("istogramma_completezza_llm.png")
plt.show()