import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator
import numpy as np

# Funkce pro kontrolu přestupného roku
def is_leap_year(year):
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

# Funkce pro přepočet na celé dny
def calculate_cas(row):
    year = row['Year']
    month = row['Month']
    day = row['Day']
    hour = row['Hour']
    minute = row['Minute']
    
    # Počet dní v každém měsíci
    days_in_months = [31, 29 if is_leap_year(year) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    
    # Kumulativní součet dní do začátku každého měsíce
    days_up_to_month = [0] + list(pd.Series(days_in_months).cumsum()[:-1])
    
    # Výpočet celkového počtu dní od začátku roku
    total_days = days_up_to_month[month - 1] + day - 31 - 29 - 11 + hour / 24 + minute / 1440
    
    return total_days


def plot_pressure_graphs(data_sets, labels, columns, data_s, data_j,orientace):
    for idx, (data, label, orient, cols) in enumerate(zip(data_sets, labels, orientace,columns)):
        # Grafy pro časy <= 0
        plt.figure(figsize=(10, 6))
        for col in cols:
            plt.plot(data[data['cas'] <= 0]['cas'], data[data['cas'] <= 0][col], label=f'{col}', linewidth=1)
        plt.xlabel('Čas [Den]')
        plt.ylabel('Tlak [kPa]')
        plt.title(f'Velikost porézního tlaku v závislosti na čase t<0, čidla {label}')
        plt.legend()
        
        # Přidání mřížky
        plt.grid(True, which='both', linestyle='--', linewidth=0.1)
        plt.gca().xaxis.set_major_locator(MultipleLocator(10))  # Popisky po 10 dnech
        plt.gca().xaxis.set_minor_locator(MultipleLocator(1))   # Vedlejší mřížka po 1 dni
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # Hlavní mřížka po 10 dnech

        plt.savefig(f'VTZ_{label}.pdf', format='pdf')
        plt.close()

        # Grafy pro časy > 0
        plt.figure(figsize=(10, 6))
        for col in cols:
            plt.plot(data[data['cas'] > 0]['cas'], data[data['cas'] > 0][col], label=f'{col}', linewidth=1)
        plt.xlabel('Čas [Den]')
        plt.ylabel('Tlak [kPa]')
        plt.title(f'Velikost porézního tlaku v závislosti na čase t>0, čidla {label}')
        plt.legend()
        
        # Přidání mřížky
        plt.grid(True, which='both', linestyle='--', linewidth=0.1)
        plt.gca().xaxis.set_major_locator(MultipleLocator(10))  # Popisky po 10 dnech
        plt.gca().xaxis.set_minor_locator(MultipleLocator(1))   # Vedlejší mřížka po 1 dni
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # Hlavní mřížka po 10 dnech

        # Přidání svislé čáry pro každý případ z data_s nebo data_j
        if orient == 'S':
            for i, cas in enumerate(data_s):
                if i == 0:
                    plt.axvline(x=cas, color='red', linestyle='--', label='Odstřel', linewidth=0.5)
                else:
                    plt.axvline(x=cas, color='red', linestyle='--', linewidth=0.5)
        elif orient == 'J':
            for i, cas in enumerate(data_j):
                if i == 0:
                    plt.axvline(x=cas, color='blue', linestyle='--', label='Odstřel', linewidth=0.5)
                else:
                    plt.axvline(x=cas, color='blue', linestyle='--', linewidth=0.5)

        plt.savefig(f'MINE_{label}.pdf', format='pdf')
        plt.close()

        print(f'Hotovy grafy pro {label}')


# Načtení souboru Konfigurace vrtu, kde je seznam listu a jejich orientace
file_path = 'konfigurace_vrtu.xlsx'
konfigurace = pd.read_excel(file_path)
labels = []
orientace = []

for _, row in konfigurace.iterrows():
    if row[0].lower() != 'rezerva':
        labels.append(row[0])
        orientace.append(row[1])

print(f'Labels: {labels}')
print(f'Orientace: {orientace}')

# Načtení nazvu Excel listu ze seznamu vrtu
file_path = 'piezo.xlsx'
data_sets = [pd.read_excel(file_path, sheet_name=label) for label in labels]
print('Data z excelovské tabulky tlaků načtena')

columns = []
for label in labels:
    df = pd.read_excel(file_path, sheet_name=label, header=None)
    col1 = df.iloc[0, 19]  # T1 je ve 20. sloupci (index 19)
    col2 = df.iloc[0, 20]  # U1 je ve 21. sloupci (index 20)
    col3 = df.iloc[0, 21]  # V1 je ve 22. sloupci (index 21)
    columns.append([col1, col2, col3])
print('Nacteny nazvy sloupcu')
print(columns)

# Načtení časů rozrazek
file_path = 'rozrazka.xlsx'
rozrazka_s = pd.read_excel(file_path, sheet_name='Sever')
data_s = rozrazka_s['cas']
rozrazka_j = pd.read_excel(file_path, sheet_name='Jih')
data_j = rozrazka_j['cas']
#print(data_s)
#print(data_j)
print('Data z excelovské tabulky času rozrazek načtena')

# Přepočet času na celé dny
for i in data_sets:
    i['cas'] = i.apply(calculate_cas, axis=1)
print('Proběhlo přepočtení hodnot časů (d,h,m) na dekadický zápis')


# Zápis výsledků do Excelu
output_file = 'output.xlsx'
with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
    for idx, df in enumerate(data_sets):
        sheet_name = f'Sheet{idx+1}'
        df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f'Data byla uložena do souboru {output_file}')


# Vytvoření grafů pomocí for cyklu
plot_pressure_graphs(data_sets, labels, columns, data_s, data_j, orientace)

