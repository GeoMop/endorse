import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator
from datetime import datetime
import numpy as np
import pathlib
script_dir = pathlib.Path(__file__).parent

# funkce v programu
# zjištění názvů vrtů a jejich orientace S/J
def read_konfigurace(file_path):
    """
    Načtení souboru Konfigurace vrtu, kde je seznam listu a jejich orientace
    """

    konfigurace = pd.read_excel(file_path)
    labels = []
    orientace = []
    col_label = 0
    col_orientace = 1

    for _, row in konfigurace.iterrows():
        label = row[col_label]
        orient = row[col_orientace]
        if label.lower() == 'rezerva':
            continue
        labels.append(label)
        orientace.append(orient)

    print(f'Labels: {labels}')
    print(f'Orientace: {orientace}')
    return labels, orientace

def read_rozrazka(file_path):
    # Načtení dat ze severní rozrážky
    rozrazka_s = pd.read_excel(file_path, sheet_name='Sever', usecols="L", skiprows=1)
    data_s = rozrazka_s.iloc[:, 0]  # Pouze první sloupec z vybraného rozsahu

    # Načtení dat z jižní rozrážky
    rozrazka_j = pd.read_excel(file_path, sheet_name='Jih', usecols="L", skiprows=1)
    data_j = rozrazka_j.iloc[:, 0]  # Pouze první sloupec z vybraného rozsahu

    print('Data ze severní rozrážky:')
    print(data_s)
    print('Data z jižní rozrážky:')
    print(data_j)
    print(f'Data z excelovské tabulky času rozrážek načtena ze souboru rozrazka_nova.xlsx')

    return data_s, data_j

# Otevře soubor piezo.xlsx, vytvoří jeho kopii piezo2.xlsx a nahradí prázdné buňky NaN.
def process_piezo_file(input_file, output_file):
    # Načtení všech listů ze souboru piezo.xlsx
    excel_data = pd.read_excel(input_file, sheet_name=None)

    # Iterace přes všechny listy a náhrada prázdných buněk NaN
    for sheet_name, df in excel_data.items():
        # Nahrazení prázdných buněk NaN
        df = df.replace(r'^\s*$', np.nan, regex=True)
        excel_data[sheet_name] = df

    # Uložení všech listů do nového souboru piezo2.xlsx
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for sheet_name, df in excel_data.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f'File {output_file} ve výstupním souboru nahrazeny prázdné hodnoty novou hodnotou NaN')

# Vytvori nove listy z listu JZ
def create_new_sheets_from_jz(file_path):
    # Načtení všech listů v souboru
    sheet_names = pd.ExcelFile(file_path).sheet_names
    
    # Kontrola, zda list 'JZ' existuje
    if 'JZ' not in sheet_names:
        raise ValueError("List 'JZ' nebyl nalezen v souboru. Dostupné listy: " + ", ".join(sheet_names))

    # Načtení dat z listu "JZ"
    df_jz = pd.read_excel(file_path, sheet_name='JZ')
    
    # Výběr sloupců pro každý nový list
    columns_1 = df_jz.loc[:, 'Logger Name':'Internal Temp(ｰC)']
    columns_2 = df_jz.loc[:, 'Sensor Reading(dg) - Channel1':'Sensor Reading(dg) - Channel4']
    columns_3 = df_jz.loc[:, 'Sensor Reading(dg) - Channel5':'Sensor Reading(kg) - Channel8']
    columns_4 = df_jz.loc[:, 'Sensor Reading(dg) - Channel9':'Sensor Reading(kg) - Channel12']
    columns_5 = df_jz.loc[:, 'Sensor Reading(dg) - Channel13':'Sensor Reading(kg) - Channel16']
    columns_6 = df_jz.loc[:, 'Sensor Temp(ｰC) - Channel1':'Sensor Temp(ｰC) - Channel4']
    columns_7 = df_jz.loc[:, 'Sensor Temp(ｰC) - Channel5':'Sensor Temp(ｰC) - Channel8']
    columns_8 = df_jz.loc[:, 'Sensor Temp(ｰC) - Channel9':'Sensor Temp(ｰC) - Channel12']
    columns_9 = df_jz.loc[:, 'Sensor Temp(ｰC) - Channel13':'Sensor Temp(ｰC) - Channel16']
    columns_10 = df_jz.loc[:, 'Array #':'datum']
    columns_22DR = df_jz.loc[:, 'tlak 14,66 m [kPa]':'tlak 8,65 m [kPa]']
    columns_23UR = df_jz.loc[:, 'tlak 10,12 m [kPa]':'tlak 3,68 m [kPa]']
    columns_24DR = df_jz.loc[:, 'tlak 9,15 m [kPa]':'tlak 2,79 m [kPa]']
    columns_26R = df_jz.loc[:, 'tlak 10,10 m [kPa]':'tlak 3,74 m [kPa]']
    
    # Vytvoření nových dataframeů kombinací sloupců
    df1 = pd.concat([columns_1, columns_2, columns_6,columns_10,columns_22DR], axis=1)
    df2 = pd.concat([columns_1, columns_3, columns_7,columns_10,columns_23UR], axis=1)
    df3 = pd.concat([columns_1, columns_4, columns_8,columns_10,columns_24DR], axis=1)
    df4 = pd.concat([columns_1, columns_5, columns_9,columns_10,columns_26R], axis=1)

    # Zápis do původního Excel souboru
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df1.to_excel(writer, sheet_name='L5-22DR', index=False)
        df2.to_excel(writer, sheet_name='L5-23UR', index=False)
        df3.to_excel(writer, sheet_name='L5-24DR', index=False)
        df4.to_excel(writer, sheet_name='L5-26R', index=False)
    
    print(f'Labels: {labels}')
    print(f"Nové listy byly přidány do souboru {file_path}")

def add_decimal_time_column(file_path, labels):
    # Definice počátečního času
    start_date = datetime(2024, 3, 11)

    # Otevření existujícího Excel souboru pro čtení a zápis
    with pd.ExcelWriter(file_path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
        for label in labels:
            # Načtení aktuálního listu
            df = pd.read_excel(file_path, sheet_name=label)
            
            # Přepočet sloupců year, month, day, hour, minute na datetime objekty
            df['datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
            
            # Výpočet dekadického času jako rozdíl v dnech od počátečního času
            df['cas'] = (df['datetime'] - start_date).dt.total_seconds() / 86400
            
            # Zápis upraveného DataFrame zpět do stejného listu
            df.drop(columns=['datetime'], inplace=True)  # Odstranění pomocného sloupce datetime
            df.to_excel(writer, sheet_name=label, index=False)

def read_measurement_data(file_path, labels):
    """
    Načtení dat z Excelu
    """
    data_sets = {label: pd.read_excel(file_path, sheet_name=label) for label in labels}
    print('Data z excelovské tabulky tlaků načtena')

    columns = []
    for label, df  in data_sets.items():
        df = pd.read_excel(file_path, sheet_name=label, header=None)
        col1 = df.iloc[0, 19]  # T1 je ve 20. sloupci (index 19)
        col2 = df.iloc[0, 20]  # U1 je ve 21. sloupci (index 20)
        col3 = df.iloc[0, 21]  # V1 je ve 22. sloupci (index 21)
        columns.append([col1, col2, col3])
    print('Nacteny nazvy sloupcu')
    print(columns)
    print(data_sets)
    return columns,data_sets

def add_new_sheet(file_path):
    # Vytvoření prázdného DataFrame
    new_data = pd.DataFrame()

    # Zápis nového listu 'vystup' do existujícího souboru
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='a') as writer:
        new_data.to_excel(writer, sheet_name='vystup', index=False)

    print(f"Nový list 'vystup' byl přidán do souboru {file_path}")

def add_unique_sorted_cas_to_vystup(file_path, labels):
    # Vytvoření prázdného DataFrame pro sloučení unikátních hodnot sloupce 'cas'
    unique_cas = pd.DataFrame()

    for label in labels:
        # Načtení listu dle názvu
        df = pd.read_excel(file_path, sheet_name=label)

        if 'cas' in df.columns:
            # Vybrání unikátních hodnot sloupce 'cas'
            cas_column = df[['cas']].drop_duplicates()
            # Sloučení s unikátními hodnotami již zpracovaných listů
            unique_cas = pd.concat([unique_cas, cas_column]).drop_duplicates()

    # Seřazení hodnot ve sloupci 'cas' vzestupně
    unique_cas = unique_cas.sort_values(by='cas')

    # Zápis výsledného DataFrame do nového listu 'vystup'
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        unique_cas.to_excel(writer, sheet_name='vystup', index=False)

    print(f"Seřazené unikátní hodnoty sloupce 'cas' byly přidány do listu 'vystup' v souboru {file_path}")

def add_date_columns_to_vystup(file_path):
    # Načtení listu 'vystup'
    df_vystup = pd.read_excel(file_path, sheet_name='vystup')

    # Definice počátečního času
    start_datetime = datetime(2024, 3, 11)

    # Přepočet 'cas' na datetime
    df_vystup['datetime'] = start_datetime + pd.to_timedelta(df_vystup['cas'], unit='D')

    # Extrakce jednotlivých složek data
    df_vystup['Year'] = df_vystup['datetime'].dt.year
    df_vystup['Month'] = df_vystup['datetime'].dt.month
    df_vystup['Day'] = df_vystup['datetime'].dt.day
    df_vystup['Hour'] = df_vystup['datetime'].dt.hour
    df_vystup['Minute'] = df_vystup['datetime'].dt.minute
    df_vystup['Seconds'] = df_vystup['datetime'].dt.second

    # Odstranění pomocného sloupce 'datetime'
    df_vystup.drop(columns=['datetime'], inplace=True)

    # Zápis zpět do listu 'vystup'
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df_vystup.to_excel(writer, sheet_name='vystup', index=False)

    print(f"Datumové sloupce byly přidány do listu 'vystup' v souboru {file_path}")
    print(df_vystup)
    return df_vystup  # Vrací upravený dataframe, pokud ho chcete dále používat


def merge_columns_to_vystup(df_vystup, file_path, labels, columns):
    # Načtení listu 'vystup'
    # df_vystup = pd.read_excel(file_path, sheet_name='vystup')

    for label, cols in zip(labels, columns):
        # Načtení listu dle názvu
        df = pd.read_excel(file_path, sheet_name=label)
        
        if 'cas' in df.columns:
            # Vybrání pouze sloupců uvedených v 'cols' a sloupce 'cas'
            df_selected = df[['cas'] + cols].copy()
            
            # Nahrazení prázdných hodnot NaN
            df_selected = df_selected.replace(r'^\s*$', np.nan, regex=True)
            
            # Sloučení s listem 'vystup' na základě sloupce 'cas'
            df_vystup = pd.merge(df_vystup, df_selected, on='cas', how='left')

    # Zápis zpět do listu 'vystup'
    with pd.ExcelWriter(df_vystup, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df_vystup.to_excel(writer, sheet_name='vystup', index=False)

    return df_vystup
    print(f"Sloupce z listů v 'Labels' byly přidány do listu 'vystup' v souboru {file_path}")

def rename_columns_in_vystup(df_vystup, labels, columns):
    # Načtení listu 'vystup'
    # df_vystup = pd.read_excel(file_path, sheet_name='vystup')

    # Vytvoření mapy pro přejmenování sloupců
    rename_map = {}
    for label, cols in zip(labels, columns):
        for col in cols:
            if 'tlak' in col:
                # Odstranění slova 'tlak' a přidání názvu listu
                new_name = f"{label} {col.replace('tlak ', '')}"
                rename_map[col] = new_name

    # Přejmenování sloupců
    df_vystup.rename(columns=rename_map, inplace=True)

    # Zápis zpět do listu 'vystup'
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df_vystup.to_excel(writer, sheet_name='vystup', index=False)
    
    df_vystup
    print(f"Názvy sloupců v listu 'vystup' byly přejmenovány podle názvu listu a upraveny.")

def log_large_differences_and_averages(df_vystup, output_minus_file, output_plus_file):
    # Seznam sloupců, které chceme porovnávat
    columns_to_check = [
        'L5-49DL 7,67 m [kPa]', 'L5-49DL 5,68 m [kPa]', 'L5-50UL 12,58 m [kPa]',
        'L5-50UL 9,72 m [kPa]', 'L5-50UL 7,72 m [kPa]', 'L5-37R 10,62 m [kPa]',
        'L5-37R 6,67 m [kPa]', 'L5-37R 3,69 m [kPa]', 'L5-26R 10,10 m [kPa]',
        'L5-26R 7,19 m [kPa]', 'L5-26R 3,74 m [kPa]', 'L5-23UR 10,12 m [kPa]',
        'L5-23UR 7,18 m [kPa]', 'L5-23UR 3,68 m [kPa]', 'L5-24DR 9,15 m [kPa]',
        'L5-24DR 6,20 m [kPa]', 'L5-24DR 2,79 m [kPa]', 'L5-22DR 14,66 m [kPa]',
        'L5-22DR 11,19 m [kPa]', 'L5-22DR 8,65 m [kPa]', 'L5-37UR 22,74 m [kPa]',
        'L5-37UR 20,75 m [kPa]', 'L5-37UR 17,67 m [kPa]'
    ]
    
    # Načtení listu 'vystup' z Excel souboru
    df = pd.read_excel(file_path, sheet_name='vystup')
    
    # Otevření obou výstupních souborů pro zápis
    with open(output_minus_file, 'w') as minus_file, open(output_plus_file, 'w') as plus_file:
        for i in range(4, len(df) - 5):
            for column in columns_to_check:
                if column in df.columns:
                    # Ignorovat NaN hodnoty
                    if pd.notna(df.loc[i, column]) and pd.notna(df.loc[i + 1, column]):
                        value_diff = abs(df.loc[i + 1, column] - df.loc[i, column])
                        
                        # Výpočet průměrů pro intervaly
                        avg_recent = df[column].iloc[i+2:i+6].dropna().mean()
                        avg_previous = df[column].iloc[i-4:i].dropna().mean()
                        
                        if not pd.isna(avg_recent) and not pd.isna(avg_previous):
                            avg_diff = abs(avg_recent - avg_previous)
                            
                            # Zkontrolovat obě podmínky
                            if value_diff > 5 and avg_diff > 5:
                                # Zkontrolovat hodnotu ve sloupci "cas"
                                output_file = minus_file if df.loc[i + 1, 'cas'] < 0 else plus_file
                                output_file.write(f"Velký rozdíl ve sloupci '{column}':\n")
                                output_file.write(f"Čas: {df.loc[i + 1, 'cas']}\n")
                                output_file.write(f"Hodnota1: {df.loc[i, column]} | Hodnota2: {df.loc[i + 1, column]}\n")
                                output_file.write(f"Rozdíl hodnot: {value_diff}\n")
                                output_file.write(f"Průměr1: {avg_previous} | Průměr2: {avg_recent}\n")
                                output_file.write(f"Rozdíl průměrů: {avg_diff}\n\n")

    print(f"Záznamy o velkých rozdílech byly uloženy do souborů {output_minus_file} a {output_plus_file}")

def log_large_differences_to_excel_and_csv(df_vystup, output_excel_file, output_csv_file, rozrazka_file):
    # Seznam sloupců, které chceme porovnávat
    columns_to_check = [
        'L5-49DL 7,67 m [kPa]', 'L5-49DL 5,68 m [kPa]', 'L5-50UL 12,58 m [kPa]',
        'L5-50UL 9,72 m [kPa]', 'L5-50UL 7,72 m [kPa]', 'L5-37R 10,62 m [kPa]',
        'L5-37R 6,67 m [kPa]', 'L5-37R 3,69 m [kPa]', 'L5-26R 10,10 m [kPa]',
        'L5-26R 7,19 m [kPa]', 'L5-26R 3,74 m [kPa]', 'L5-23UR 10,12 m [kPa]',
        'L5-23UR 7,18 m [kPa]', 'L5-23UR 3,68 m [kPa]', 'L5-24DR 9,15 m [kPa]',
        'L5-24DR 6,20 m [kPa]', 'L5-24DR 2,79 m [kPa]', 'L5-22DR 14,66 m [kPa]',
        'L5-22DR 11,19 m [kPa]', 'L5-22DR 8,65 m [kPa]', 'L5-37UR 22,74 m [kPa]',
        'L5-37UR 20,75 m [kPa]', 'L5-37UR 17,67 m [kPa]'
    ]
    
    # Načtení listu 'vystup' z Excel souboru
    df = pd.read_excel(file_path, sheet_name='vystup')
    
    # Načtení časů rozrážek
    rozrazka_s = pd.read_excel(rozrazka_file, sheet_name='Sever', usecols="L", skiprows=1)
    rozrazka_j = pd.read_excel(rozrazka_file, sheet_name='Jih', usecols="L", skiprows=1)
    data_s = rozrazka_s.iloc[:, 0]
    data_j = rozrazka_j.iloc[:, 0]
    
    # Data pro zápis
    output_minus = []
    output_plus = []
    
    for i in range(4, len(df) - 5):
        for column in columns_to_check:
            if column in df.columns:
                j = i
                while j >= 0 and pd.isna(df.loc[j, column]):
                    j -= 1
                
                if j >= 0 and pd.notna(df.loc[j, column]) and pd.notna(df.loc[i + 1, column]):
                    value_diff = abs(df.loc[i + 1, column] - df.loc[j, column])
                    
                    avg_recent = df[column].iloc[i+2:i+6].dropna().mean()
                    avg_previous = df[column].iloc[max(0, j-4):j].dropna().mean()
                    
                    if not pd.isna(avg_recent) and not pd.isna(avg_previous):
                        avg_diff = abs(avg_recent - avg_previous)
                        
                        if value_diff > 3 and avg_diff > 3:
                            output_data = {
                                'Čas': df.loc[i + 1, 'cas'],
                                'Year': df.loc[i + 1, 'Year'],
                                'Month': df.loc[i + 1, 'Month'],
                                'Day': df.loc[i + 1, 'Day'],
                                'Hour': df.loc[i + 1, 'Hour'],
                                'Minute': df.loc[i + 1, 'Minute'],
                                'Seconds': df.loc[i + 1, 'Seconds'],
                                'Čidlo': column,
                                'Tlak před': df.loc[j, column],
                                'Tlak po': df.loc[i + 1, column],
                                'Rozdíl hodnot': value_diff,
                                'Průměr před': avg_previous,
                                'Průměr po': avg_recent,
                                'Rozdíl průměrů': avg_diff
                            }
                            if df.loc[i + 1, 'cas'] < 0:
                                output_minus.append(output_data)
                            else:
                                output_plus.append(output_data)
    
    # Přidání sloupce "činnost"
    df['činnost'] = df['cas'].apply(lambda x: 'razba' if any((-0.1 <= x - cas <= 0.3) for cas in data_s) or any((-0.1 <= x - cas <= 0.3) for cas in data_j) else 'nevime')
    
    # Uložení do Excelu a CSV
    with pd.ExcelWriter(output_excel_file, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Output', index=False)
    
    df.to_csv(output_csv_file, index=False, sep=';')
    
    print(f"Výsledky byly zapsány do Excel souboru {output_excel_file} a CSV souboru {output_csv_file}")


#hlavni program
# otevře soubor konfigurace_vrtu a zjistí názvy vrtů a jejich orientaci S/J
labels, orientace = read_konfigurace(script_dir/'konfigurace_vrtu.xlsx')

# nacte casy rozrazek, otevře soubor rozrazka
data_s, data_j = read_rozrazka(script_dir/'rozrazka_nova.xlsx')

# zkopírování souboru na piezo2.xlsx a nahrazením prázdných buněk hodnotou NaN
input_file = script_dir / 'piezo.xlsx'
output_file = script_dir / 'piezo2.xlsx'
process_piezo_file(input_file, output_file)

# Načtení nazvu Excel listu ze seznamu vrtu
file_path = script_dir /'piezo2.xlsx'
sheet_names = pd.ExcelFile(file_path).sheet_names
print(f"Dostupné listy: " + ", ".join(sheet_names))

# Vytvori nove listy z listu JZ
create_new_sheets_from_jz(file_path)

# Přidání sloupce s decimálním časem
add_decimal_time_column(file_path, labels)
columns,data_sets = read_measurement_data(file_path, labels)

add_new_sheet(file_path)
add_unique_sorted_cas_to_vystup(file_path, labels)
df_vystup = add_date_columns_to_vystup(file_path)
df_vystup = merge_columns_to_vystup(df_vystup, file_path, labels, columns)
df_vystup = rename_columns_in_vystup(df_vystup, labels, columns)

# Nalezeni rozdilu mezi radky
#log_large_differences_to_excel_and_csv(df_vystup, script_dir / 'output_results.xlsx', script_dir / 'output_results.csv', script_dir / 'rozrazka_nova.xlsx')
#log_large_differences_and_averages(script_dir /'piezo2.xlsx', script_dir /'output_minus.txt', script_dir /'output_plus.txt')
#log_large_differences_and_averages(df_vystup, script_dir / 'output_minus.txt', script_dir / 'output_plus.txt', script_dir / 'output_results.xlsx')