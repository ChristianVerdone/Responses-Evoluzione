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
palette = sns.color_palette("husl", len(df_mean_completezza))  # Use a different color palette
sns.barplot(x='LLM', y='Completezza', data=df_mean_completezza, palette=palette)
plt.title("Average Completeness Value for Each LLM", fontsize=16)
plt.ylabel("Completeness")
plt.xlabel("LLM")
plt.xticks(rotation=45)
plt.tight_layout()

# Save or show the plot
plt.savefig("completeness_histogram_llm.png")
plt.show()