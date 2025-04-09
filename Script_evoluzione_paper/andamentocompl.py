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

# Grafico dell'andamento della completezza per ciascun LLM
plt.figure(figsize=(15, 8))
sns.lineplot(x='Risposta', y='Completezza', hue='LLM', data=df_completezza, marker='o')

# Add information to the chart
plt.title("Trend of Completeness for Each LLM", fontsize=16)
plt.ylabel("Completeness")
plt.xlabel("Response")
plt.legend(title="LLM")
plt.xticks(range(1, 11))  # Assuming there are 10 responses
plt.tight_layout()

# Salva o mostra il grafico
plt.savefig("andamento_completezza_llm.png")
plt.show()