import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carica il file Excel (converti in Excel se necessario)
file_path = r"C:\Users\marco\Desktop\RisposteEvoluzione.xlsx"  # Sostituisci con il nome del file
df = pd.read_excel(file_path)

# Visualizza un'anteprima dei dati
print(df.head())

# Pulisci e normalizza i dati (se necessario)
df['Accuratezza'] = pd.to_numeric(df['Accuratezza'], errors='coerce')
df['Completezza'] = pd.to_numeric(df['Completezza'], errors='coerce')

# Creare un DataFrame lungo per sns.barplot con hue
df_long = pd.melt(df, id_vars=['LLM'], value_vars=['Accuratezza', 'Completezza'], 
                  var_name='Metrica', value_name='Valore')

# Grafico di Accuratezza e Completezza per ciascun LLM
plt.figure(figsize=(12, 6))
sns.barplot(x='LLM', y='Valore', hue='Metrica', data=df_long, ci=None)

# Add information to the chart
plt.title("Accuracy and Completeness for LLM", fontsize=16)
plt.ylabel("Value")
plt.xlabel("LLM")
plt.legend(title="Metric")
plt.xticks(rotation=45)
plt.tight_layout()

# Salva o mostra il grafico
plt.savefig("metriche_llm.png")
plt.show()

# Filtra e crea un grafico per ciascun LLM
llms = df['LLM'].unique()

for llm in llms:
    df_llm = df[df['LLM'] == llm]

    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Risposta-timestamp', y='Accuratezza', data=df_llm, marker='o', label='Accuratezza')
    sns.lineplot(x='Risposta-timestamp', y='Completezza', data=df_llm, marker='o', label='Completezza')

    # Aggiungi informazioni al grafico
    plt.title(f"Accuratezza e Completezza per ciascuna Risposta - {llm}", fontsize=16)
    plt.ylabel("Valore")
    plt.xlabel("Risposta-timestamp")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Salva o mostra il grafico
    plt.savefig(f"metriche_risposte_{llm}.png")
    plt.show()