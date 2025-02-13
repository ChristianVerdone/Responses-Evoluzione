import pandas as pd

# Carica il file Excel
file_path = r".\RisposteLLMEvoluzione.xlsx"  # Sostituisci con il nome del file
df = pd.read_excel(file_path)

# Pulisci e normalizza i dati (se necessario)
df['Accuratezza'] = pd.to_numeric(df['Accuratezza'], errors='coerce')
df['Completezza'] = pd.to_numeric(df['Completezza'], errors='coerce')

# Calcola i valori medi di Accuratezza e Completezza per ciascun LLM
mean_values = df.groupby('LLM')[['Accuratezza', 'Completezza']].mean()

# Stampa i valori medi per ciascun LLM
for llm, values in mean_values.iterrows():
    print(f"LLM: {llm}")
    print(f"  Accuratezza media: {values['Accuratezza']:.2f}")
    print(f"  Completezza media: {values['Completezza']:.2f}")
    print()
