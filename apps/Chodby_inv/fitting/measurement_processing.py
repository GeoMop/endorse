import yaml
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator
from datetime import datetime
import numpy as np
import pathlib
import os

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

def read_vstupy(file_path):
    """
    Načte hodnoty tmin, tmax a zapis_do_souboru ze souboru vstup.yaml.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    
    tmin = data.get("tmin")
    tmax = data.get("tmax")
    zapis_do_souboru = data.get("zapis_do_souboru", False)
    v_diff = data.get("value_diff")
    a_diff = data.get("avg_diff")

    if tmin is None or tmax is None or v_diff is None or a_diff is None:
        raise ValueError("Soubor neobsahuje platné hodnoty tmin, tmax, value_diff a avg_diff.")
    
    print(f"Načteno: tmin={tmin}, tmax={tmax}, zapis_do_souboru={zapis_do_souboru}, v_diff={v_diff}, a_diff={a_diff}")
    return tmin, tmax, zapis_do_souboru, v_diff, a_diff

def process_piezo_file(input_file, output_file=None, save_to_excel=True):
    """
    Načte všechny listy ze souboru piezo.xlsx, nahradí prázdné hodnoty NaN 
    a volitelně uloží výsledek do Excelu.

    Args:
        input_file (str or Path): Cesta k vstupnímu souboru.
        output_file (str or Path, optional): Cesta k výstupnímu souboru. Výchozí None.
        save_to_excel (bool, optional): Pokud True, uloží výstup do Excelu. Výchozí True.

    Returns:
        dict: Slovník obsahující DataFrame pro každý list.
    """
    excel_data = pd.read_excel(input_file, sheet_name=None)

    # Iterace přes všechny listy a náhrada prázdných buněk NaN
    for sheet_name, df in excel_data.items():
        df = df.replace(r'^\s*$', np.nan, regex=True)
        excel_data[sheet_name] = df

    print(f'Data z "{input_file}" byla načtena a prázdné buňky nahrazeny NaN.')

    # Uložení do Excelu pouze pokud je zapnuto `save_to_excel`
    if save_to_excel and output_file:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            for sheet_name, df in excel_data.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f'Výstup uložen do "{output_file}".')

    return excel_data

def create_new_sheets_from_jz(data_frames, output_file=None, save_to_excel=True):
    """
    Vytvoří nové listy z listu 'JZ', rozšíří data_frames a uloží do souboru piezo4.xlsx.

    Args:
        data_frames (dict): Slovník obsahující všechny listy jako DataFrame.
        output_file (str): Cesta k výstupnímu Excel souboru (výchozí 'piezo4.xlsx').

    Returns:
        dict: Aktualizovaný slovník s přidanými listy.
    """
    # Kontrola, zda 'JZ' existuje v datech
    if 'JZ' not in data_frames:
        raise ValueError("List 'JZ' nebyl nalezen v datech.")

    df_jz = data_frames['JZ']  # Načtení listu 'JZ'
    
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
    data_frames['L5-22DR'] = pd.concat([columns_1, columns_2, columns_6, columns_10, columns_22DR], axis=1)
    data_frames['L5-23UR'] = pd.concat([columns_1, columns_3, columns_7, columns_10, columns_23UR], axis=1)
    data_frames['L5-24DR'] = pd.concat([columns_1, columns_4, columns_8, columns_10, columns_24DR], axis=1)
    data_frames['L5-26R'] = pd.concat([columns_1, columns_5, columns_9, columns_10, columns_26R], axis=1)

    # Uložení do Excelu

    if save_to_excel and output_file:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            for sheet_name, df in data_frames.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"Výstup uložen do {output_file}.")
    
    print(f"Nové listy byly přidány do data_frames a uloženy do souboru {output_file}")

    # Výpis všech listů po přidání nových
    print("Listy po rozšíření:", list(data_frames.keys()))

    return data_frames  # Vrací aktualizovaný slovník


def add_decimal_time_column(data_frames, labels, output_file="piezo5.xlsx", save_to_excel=True):
    """
    Přidá sloupec s dekadickým časem do listů v data_frames a uloží výstup do souboru piezo5.xlsx.

    Args:
        data_frames (dict): Slovník obsahující DataFrame pro každý list.
        labels (list): Seznam listů, které mají být upraveny.
        output_file (str): Cesta k výstupnímu souboru (výchozí "piezo5.xlsx").

    Returns:
        dict: Aktualizovaný slovník data_frames s přidaným sloupcem "cas".
    """
    # Definice počátečního času
    start_date = datetime(2024, 3, 11)

    # Iterace přes všechny listy a úprava dat
    for label in labels:
        if label in data_frames:
            df = data_frames[label]

            # Ověření, že obsahuje potřebné sloupce
            required_columns = ['Year', 'Month', 'Day', 'Hour', 'Minute']
            if all(col in df.columns for col in required_columns):
                # Přepočet sloupců Year, Month, Day, Hour, Minute na datetime
                df['datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
                
                # Výpočet dekadického času
                df['cas'] = (df['datetime'] - start_date).dt.total_seconds() / 86400
                
                # Odstranění pomocného sloupce datetime
                df.drop(columns=['datetime'], inplace=True)
                
                # Uložení zpět do slovníku
                data_frames[label] = df
            else:
                print(f"Varování: List '{label}' neobsahuje všechny potřebné sloupce.")

    # Uložení upravených dat do Excelu
    if save_to_excel and output_file:   
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            for sheet_name, df in data_frames.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Sloupec 'cas' byl přidán do listů a výstup byl uložen do {output_file}.")

    # Výpis prvních řádků upraveného listu pro kontrolu
    print(data_frames["JZ"].head())

    return data_frames  # Vrací aktualizovaný slovník

def read_measurement_data(final_data_frames, labels):
    """
    Načte data z `final_data_frames` a extrahuje názvy sloupců.

    Args:
        final_data_frames (dict): Slovník obsahující DataFrame pro každý list.
        labels (list): Seznam listů, ze kterých se mají extrahovat sloupce.

    Returns:
        tuple: (columns, data_sets)
        - columns (list): Seznam názvů tří sloupců pro každý list.
        - data_sets (dict): Slovník s načtenými DataFrame.
    """
    data_sets = {label: final_data_frames[label] for label in labels if label in final_data_frames}
    print('Data z dataframe načtena.')

    columns = []
    for label, df in data_sets.items():
        # Ověření, že DataFrame má dostatek sloupců
        if df.shape[1] > 21:  
            col1 = df.columns[19]  # Správný název sloupce
            col2 = df.columns[20]
            col3 = df.columns[21]
            columns.append([col1, col2, col3])
        else:
            print(f"Varování: List '{label}' nemá dostatek sloupců.")

    print('Nacteny nazvy sloupcu:', columns)
    return columns, data_sets


def filter_data_by_time_range(data_frames, labels, tmin, tmax, output_file="piezo6.xlsx", save_to_excel=True):
    """
    Filtrovat pouze listy uvedené v `labels` podle časového intervalu (tmin, tmax)
    a uložit výsledek do piezo6.xlsx.

    Args:
        data_frames (dict): Slovník obsahující DataFrame pro každý list.
        labels (list): Seznam listů, které mají být filtrovány.
        tmin (float): Dolní mez filtru.
        tmax (float): Horní mez filtru.
        output_file (str): Cesta k výstupnímu souboru (výchozí "piezo6.xlsx").

    Returns:
        dict: Aktualizovaný slovník `data_frames` s filtrovanými daty pro listy v `labels`.
    """
    filtered_data_frames = {}

    for sheet_name, df in data_frames.items():
        if sheet_name in labels:  # Pouze listy v labels
            if 'cas' in df.columns:
                df_filtered = df[(df['cas'] >= tmin) & (df['cas'] <= tmax)].copy()
                filtered_data_frames[sheet_name] = df_filtered
            else:
                print(f"Varování: List '{sheet_name}' neobsahuje sloupec 'cas'. Nebyl filtrován.")
                filtered_data_frames[sheet_name] = df  # Ponecháme původní verzi listu
        else:
            # List není v labels, ponecháme ho beze změny
            filtered_data_frames[sheet_name] = df

    # Uložení filtrovaných dat do Excelu piezo6.xlsx
    if save_to_excel and output_file: 
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            for sheet_name, df in filtered_data_frames.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Listy {labels} byly filtrovány podle intervalu ({tmin}, {tmax}) a uloženy do {output_file}.")
    return filtered_data_frames  # Vrací slovník s filtrovanými daty

def remove_columns_from_labels(data_frames, labels, output_file="piezo7.xlsx", save_to_excel=True):
    """
    Odstraní sloupce 7 až 19 z listů uvedených v `labels` a uloží výsledek do piezo7.xlsx.

    Args:
        data_frames (dict): Slovník obsahující DataFrame pro každý list.
        labels (list): Seznam listů, které mají být upraveny.
        output_file (str): Cesta k výstupnímu souboru (výchozí "piezo7.xlsx").

    Returns:
        dict: Aktualizovaný slovník `data_frames` s upravenými listy.
    """
    updated_data_frames = {}

    for sheet_name, df in data_frames.items():
        if sheet_name in labels:  # Pouze pro listy v labels
            if df.shape[1] > 19:  # Ověření, že je dostatek sloupců k odstranění
                df = df.drop(df.columns[7:19], axis=1)  # Odstranění sloupců 7 až 19
                print(f"Sloupce 7 až 19 byly odstraněny v listu '{sheet_name}'.")
            else:
                print(f"Varování: List '{sheet_name}' nemá dostatek sloupců pro odstranění.")
        updated_data_frames[sheet_name] = df  # Uložíme zpět do slovníku

    # Uložení do Excelu piezo7.xlsx
    if save_to_excel and output_file:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            for sheet_name, df in updated_data_frames.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Upravené listy byly uloženy do {output_file}.")
    return updated_data_frames  # Vrací aktualizovaný slovník

def add_new_sheet(final_data_frames):
    """
    Přidá nový prázdný list 'vystup' do final_data_frames.

    Args:
        final_data_frames (dict): Slovník obsahující DataFrame pro každý list.

    Returns:
        dict: Aktualizovaný slovník final_data_frames.
    """
    final_data_frames["vystup"] = pd.DataFrame()  # Vytvoření prázdného DataFrame
    print("Nový list 'vystup' byl přidán do final_data_frames.")
    return final_data_frames

def add_unique_sorted_cas_to_vystup(final_data_frames, labels, output_file="piezo8.xlsx", save_to_excel=True):
    """
    Shromáždí unikátní hodnoty 'cas' z listů v labels, seřadí je a uloží do listu 'vystup'.
    Výsledek se zároveň uloží do souboru 'piezo8.xlsx'.

    Args:
        final_data_frames (dict): Slovník obsahující DataFrame pro každý list.
        labels (list): Seznam listů, ze kterých se získají unikátní hodnoty 'cas'.
        output_file (str): Cesta k výstupnímu souboru (výchozí "piezo8.xlsx").

    Returns:
        dict: Aktualizovaný slovník final_data_frames s novým listem 'vystup'.
    """
    unique_cas = pd.DataFrame()  # Vytvoření prázdného DataFrame pro unikátní hodnoty cas

    for label in labels:
        if label in final_data_frames:
            df = final_data_frames[label]

            if 'cas' in df.columns:
                cas_column = df[['cas']].drop_duplicates()
                unique_cas = pd.concat([unique_cas, cas_column]).drop_duplicates()

    # Seřazení hodnot ve sloupci 'cas' vzestupně
    unique_cas = unique_cas.sort_values(by='cas')

    # Uložení do final_data_frames
    final_data_frames["vystup"] = unique_cas
    print("Seřazené unikátní hodnoty 'cas' byly přidány do final_data_frames.")

    # Uložení výsledku do Excelu piezo8.xlsx
    if save_to_excel and output_file:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            for sheet_name, df in final_data_frames.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Všechna data byla uložena do {output_file}.")

    return final_data_frames

def add_date_columns_to_vystup(final_data_frames, output_file="piezo9.xlsx", save_to_excel=True):
    """
    Přidá časové sloupce (Year, Month, Day, Hour, Minute, Seconds) do listu 'vystup'.
    Výsledek uloží do souboru piezo9.xlsx.

    Args:
        final_data_frames (dict): Slovník obsahující DataFrame pro každý list.
        output_file (str): Název výstupního souboru (výchozí "piezo9.xlsx").

    Returns:
        dict: Aktualizovaný slovník final_data_frames.
    """
    if "vystup" not in final_data_frames:
        raise ValueError("List 'vystup' neexistuje v final_data_frames.")

    df_vystup = final_data_frames["vystup"].copy()

    # Definice počátečního času
    start_datetime = datetime(2024, 3, 11)

    # Přepočet 'cas' na datetime
    df_vystup['datetime'] = start_datetime + pd.to_timedelta(df_vystup['cas'], unit='D')

    # Extrakce jednotlivých složek času
    df_vystup['Year'] = df_vystup['datetime'].dt.year
    df_vystup['Month'] = df_vystup['datetime'].dt.month
    df_vystup['Day'] = df_vystup['datetime'].dt.day
    df_vystup['Hour'] = df_vystup['datetime'].dt.hour
    df_vystup['Minute'] = df_vystup['datetime'].dt.minute
    df_vystup['Seconds'] = df_vystup['datetime'].dt.second

    # Odstranění pomocného sloupce 'datetime'
    df_vystup.drop(columns=['datetime'], inplace=True)

    # Uložení zpět do final_data_frames
    final_data_frames["vystup"] = df_vystup

    print("Časové sloupce byly přidány do 'vystup'.")

    # Uložení do Excelu
    if save_to_excel and output_file:
    	with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            for sheet_name, df in final_data_frames.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Výstup byl uložen do {output_file}.")
    return final_data_frames

def merge_columns_to_vystup(final_data_frames, labels, columns, output_file="piezo10.xlsx", save_to_excel=True):
    """
    Sloučí vybrané sloupce z listů v `labels` do listu 'vystup' na základě sloupce 'cas'.
    Výsledek uloží do souboru piezo10.xlsx.

    Args:
        final_data_frames (dict): Slovník obsahující DataFrame pro každý list.
        labels (list): Seznam listů, ze kterých se mají sloupce přidat.
        columns (list of list): Seznam sloupců, které se mají přidat.
        output_file (str): Název výstupního souboru (výchozí "piezo10.xlsx").

    Returns:
        dict: Aktualizovaný slovník final_data_frames.
    """
    if "vystup" not in final_data_frames:
        raise ValueError("List 'vystup' neexistuje v final_data_frames.")

    df_vystup = final_data_frames["vystup"].copy()

    for label, cols in zip(labels, columns):
        if label in final_data_frames:
            df = final_data_frames[label]

            if 'cas' in df.columns:
                df_selected = df[['cas'] + cols].copy()
                df_selected = df_selected.replace(r'^\s*$', np.nan, regex=True)

                # Sloučení s vystup na základě 'cas'
                df_vystup = pd.merge(df_vystup, df_selected, on='cas', how='left')

    # Uložení zpět do final_data_frames
    final_data_frames["vystup"] = df_vystup

    print(f"Sloupce z listů {labels} byly přidány do 'vystup'.")

    # Uložení do Excelu
    if save_to_excel and output_file:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            for sheet_name, df in final_data_frames.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Výstup byl uložen do {output_file}.")
    return final_data_frames

def rename_columns_in_vystup(final_data_frames, labels, columns, output_file="piezo11.xlsx", save_to_excel=True):
    """
    Přejmenuje sloupce v listu 'vystup' na základě názvu listu.
    Výsledek uloží do souboru piezo11.xlsx.

    Args:
        final_data_frames (dict): Slovník obsahující DataFrame pro každý list.
        labels (list): Seznam listů, jejichž sloupce se přejmenují.
        columns (list of list): Seznam sloupců, které budou přejmenovány.
        output_file (str): Název výstupního souboru (výchozí "piezo11.xlsx").

    Returns:
        dict: Aktualizovaný slovník final_data_frames.
    """
    if "vystup" not in final_data_frames:
        raise ValueError("List 'vystup' neexistuje v final_data_frames.")

    df_vystup = final_data_frames["vystup"].copy()

    rename_map = {}
    for label, cols in zip(labels, columns):
        for col in cols:
            if 'tlak' in col:
                new_name = f"{label} {col.replace('tlak ', '')}"
                rename_map[col] = new_name

    # Přejmenování sloupců
    df_vystup.rename(columns=rename_map, inplace=True)

    # Uložení zpět do final_data_frames
    final_data_frames["vystup"] = df_vystup

    print("Sloupce v 'vystup' byly přejmenovány.")

    # Uložení do Excelu
    if save_to_excel and output_file:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            for sheet_name, df in final_data_frames.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Výstup byl uložen do {output_file}.")
    return final_data_frames

def keep_only_vystup(final_data_frames, output_file="piezo12.xlsx", save_to_excel=True):
    """
    Zachová pouze list 'vystup' v final_data_frames a odstraní všechny ostatní.
    Výsledek uloží do souboru piezo13.xlsx.

    Args:
        final_data_frames (dict): Slovník obsahující DataFrame pro každý list.
        output_file (str): Název výstupního souboru (výchozí "piezo13.xlsx").

    Returns:
        dict: Aktualizovaný slovník final_data_frames obsahující pouze list 'vystup'.
    """
    if "vystup" not in final_data_frames:
        raise ValueError("List 'vystup' neexistuje v final_data_frames.")

    # Zachování pouze listu 'vystup'
    final_data_frames = {"vystup": final_data_frames["vystup"].copy()}

    print("Všechny listy kromě 'vystup' byly odstraněny.")

    # Uložení do Excelu
    if save_to_excel and output_file:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            final_data_frames["vystup"].to_excel(writer, sheet_name="vystup", index=False)

    print(f"Výstup byl uložen do {output_file}.")
    return final_data_frames

def plot_pressure_graphs(final_data_frames, labels, columns, data_s, data_j, orientace, tmin, tmax, script_dir):
    """
    Vytvoří a uloží jediný graf pro každý list v intervalu tmin až tmax do složky `script_dir`.
    Pokud jsou hodnoty `data_s` nebo `data_j` mimo interval [tmin, tmax], ignorují se.

    Args:
        final_data_frames (dict): Slovník obsahující DataFrame pro každý list.
        labels (list): Seznam listů, které mají být zpracovány.
        columns (list of list): Seznam sloupců, které mají být vykresleny.
        data_s (list): Seznam časů pro jižní orientaci.
        data_j (list): Seznam časů pro severní orientaci.
        orientace (list): Seznam orientací ('S' nebo 'J') pro jednotlivé listy.
        tmin (float): Dolní hranice vykresleného intervalu.
        tmax (float): Horní hranice vykresleného intervalu.
        script_dir (str): Cesta k výstupní složce pro uložení grafů.
    """

    # Vytvoření složky, pokud neexistuje
    os.makedirs(script_dir, exist_ok=True)

    for idx, (label, orient, cols) in enumerate(zip(labels, orientace, columns)):
        if label in final_data_frames:
            data = final_data_frames[label]

            if isinstance(data, pd.DataFrame) and 'cas' in data.columns:
                # Filtrace dat pouze v intervalu tmin až tmax
                subset = data[(data['cas'] >= tmin) & (data['cas'] <= tmax)]

                # Vytvoření grafu
                plt.figure(figsize=(10, 6))
                for col in cols:
                    if col in subset.columns:
                        plt.plot(subset['cas'], subset[col], label=f'{col}', linewidth=1)
                    else:
                        print(f"Sloupec {col} není v datech pro {label}.")

                plt.xlabel('Čas [Den]')
                plt.ylabel('Tlak [kPa]')
                plt.title(f'Velikost porézního tlaku v závislosti na čase (t = {tmin} až t = {tmax}), čidla {label}')
                plt.legend()
                plt.grid(True, which='both', linestyle='--', linewidth=0.1)
                plt.gca().xaxis.set_major_locator(MultipleLocator(10))
                plt.gca().xaxis.set_minor_locator(MultipleLocator(1))
                plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
                plt.xlim(tmin, tmax)  # Nastavení rozsahu vodorovné osy

                # Přidání vertikálních čar pro odstřely pouze v rozsahu [tmin, tmax]
                if orient == 'S':
                    for i, cas in enumerate(data_s):
                        if tmin <= cas <= tmax:
                            plt.axvline(x=cas, color='red', linestyle='--', label='Odstřel' if i == 0 else "", linewidth=0.5)
                elif orient == 'J':
                    for i, cas in enumerate(data_j):
                        if tmin <= cas <= tmax:
                            plt.axvline(x=cas, color='blue', linestyle='--', label='Odstřel' if i == 0 else "", linewidth=0.5)

                # Sestavení názvu souboru a uložení do script_dir
                graph_filename = os.path.join(script_dir, f'PRESSURE_{label}.pdf')
                plt.savefig(graph_filename, format='pdf')
                plt.close()

                print(f'Hotový graf pro {label} v intervalu {tmin} až {tmax}, uložen do {graph_filename}')
            else:
                print(f"Sloupec 'cas' chybí v datech pro {label}.")
        else:
            print(f"Chyba: Data pro {label} nejsou dostupná v final_data_frames.")

def log_large_differences(final_data_frames, v_diff, a_diff, rozrazka_file, output_excel_file, output_csv_file):
    """
    Analyzuje rozdíly v tlaku mezi řádky v `final_data_frames` a zapisuje pouze významné rozdíly do Excelu a CSV.

    Args:
        final_data_frames (dict): Slovník obsahující DataFrame pro každý list.
        v_diff (float): Minimální hodnota rozdílu pro zápis.
        a_diff (float): Minimální hodnota rozdílu průměrů pro zápis.
        rozrazka_file (str or Path): Cesta k souboru s daty rozrážek.
        output_excel_file (str or Path): Cesta k výstupnímu Excel souboru.
        output_csv_file (str or Path): Cesta k výstupnímu CSV souboru.

    Returns:
        dict: Aktualizovaný `final_data_frames` s přidanými hodnotami.
    """
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
    
    # Ověření existence listu "vystup" v datech
    if "vystup" not in final_data_frames:
        raise ValueError("❌ Chyba: List 'vystup' neexistuje v `final_data_frames`.")

    df = final_data_frames["vystup"]

    # Načtení časů rozrážek
    rozrazka_s = pd.read_excel(rozrazka_file, sheet_name='Sever', usecols="L", skiprows=1).dropna()
    rozrazka_j = pd.read_excel(rozrazka_file, sheet_name='Jih', usecols="L", skiprows=1).dropna()
    data_s = rozrazka_s.iloc[:, 0].tolist()
    data_j = rozrazka_j.iloc[:, 0].tolist()
    all_rozrazka_times = data_s + data_j

    # Data pro zápis
    output_minus = []
    output_plus = []
    last_shot_time = {col: None for col in columns_to_check}
    
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
                        
                        if value_diff > v_diff and avg_diff > a_diff:
                            current_time = df.loc[i + 1, 'cas']
                            if last_shot_time[column] is None and any(abs(current_time - t) <= 0.02 for t in all_rozrazka_times):
                                change_type = "střílení"
                                last_shot_time[column] = current_time
                            elif last_shot_time[column] is not None and current_time - last_shot_time[column] > 0 and current_time - last_shot_time[column] <= 0.05:
                                change_type = "reakce_na_střílení"
                            else:
                                change_type = "nevysvětleno"
                                last_shot_time[column] = None
                            
                            output_data = {
                                'Čas': current_time,
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
                                'Rozdíl průměrů': avg_diff,
                                'Druh změny': change_type
                            }
                            if current_time < 0:
                                output_minus.append(output_data)
                            else:
                                output_plus.append(output_data)
    
    # Vytvoření DataFrame pouze s relevantními daty
    df_output = pd.DataFrame(output_minus + output_plus)
    
    # Pokud jsou nějaká data k uložení, zapíšeme je
    if not df_output.empty:
        with pd.ExcelWriter(output_excel_file, engine='openpyxl') as writer:
            df_output.to_excel(writer, sheet_name='Filtered_Output', index=False)
        
        df_output.to_csv(output_csv_file, index=False, sep=';')
        
        print(f"✅ Filtrované výsledky byly uloženy:\n   📂 Excel: {output_excel_file}\n   📂 CSV: {output_csv_file}")
    else:
        print("⚠️ Nebyly nalezeny žádné významné rozdíly k uložení.")
    
    return final_data_frames




# Hlavni program - načtení vstupních dat
# Otevře soubor konfigurace_vrtu a zjistí názvy vrtů a jejich orientaci S/J
labels, orientace = read_konfigurace(script_dir/'konfigurace_vrtu.xlsx')
# Nacte casy rozrazek, otevře soubor rozrazka
data_s, data_j = read_rozrazka(script_dir/'rozrazka_nova.xlsx')
# Nacte minimální a maximální čas, pro který bude úloha zpracovávána a zda bude zapisovat do excelu
tmin, tmax, zapis_do_souboru, v_diff, a_diff = read_vstupy(script_dir / 'vstup.yaml')


# Hlavni program - úprava dat
# Načtení dat do paměti a zároveň uložení do "piezo3.xlsx"
data_frames = process_piezo_file(
    script_dir / "piezo.xlsx",
    output_file=script_dir / "piezo3.xlsx" if zapis_do_souboru else None,
    save_to_excel=zapis_do_souboru
)

# Vytvoříme nové listy z 'JZ' a uložíme do piezo4.xlsx
data_frames = create_new_sheets_from_jz(
    data_frames, 
    output_file=script_dir / "piezo4.xlsx" if zapis_do_souboru else None, 
    save_to_excel=zapis_do_souboru
)

# Přidáme dekadický čas do vybraných listů a uložíme do piezo5.xlsx
# data_frames = add_decimal_time_column(data_frames, labels, output_file=script_dir/"piezo5.xlsx")
data_frames = add_decimal_time_column(
    data_frames, labels, 
    output_file=script_dir / "piezo5.xlsx" if zapis_do_souboru else None, 
    save_to_excel=zapis_do_souboru
)

# Filtrování dat a uložení výsledku do piezo6.xlsx
# filtered_data_frames = filter_data_by_time_range(data_frames, labels, tmin, tmax, output_file=script_dir/"piezo6.xlsx")
filtered_data_frames = filter_data_by_time_range(
    data_frames, labels, tmin, tmax, 
    output_file=script_dir / "piezo6.xlsx" if zapis_do_souboru else None, 
    save_to_excel=zapis_do_souboru
)

columns, data_sets = read_measurement_data(filtered_data_frames, labels)

# Odstranění sloupců 7 až 19 a uložení výsledku do piezo7.xlsx
final_data_frames = remove_columns_from_labels(
    filtered_data_frames, labels, 
    output_file=script_dir / "piezo7.xlsx" if zapis_do_souboru else None, 
    save_to_excel=zapis_do_souboru
)

# Přidání unikátních hodnot 'cas' do listu 'vystup' a uložení do "piezo8.xlsx"
final_data_frames = add_unique_sorted_cas_to_vystup(
    final_data_frames, labels, 
    output_file=script_dir / "piezo8.xlsx" if zapis_do_souboru else None, 
    save_to_excel=zapis_do_souboru
)
# Výpočet "year", "month", "day", "hour, z načtených dekadických časů a zápis do souboru "piezo9.xlsx". 
final_data_frames = add_date_columns_to_vystup(
    final_data_frames, 
    output_file=script_dir / "piezo9.xlsx" if zapis_do_souboru else None, 
    save_to_excel=zapis_do_souboru
)

# Zapíše tlaky příšlušným čidlům. Prázdné data jsou nahrazeny NaN 
final_data_frames = merge_columns_to_vystup(
    final_data_frames, labels, columns, 
    output_file=script_dir / "piezo10.xlsx" if zapis_do_souboru else None, 
    save_to_excel=zapis_do_souboru
)

# Přejmenuje hlavičky podle názvu vrtu 
final_data_frames = rename_columns_in_vystup(
    final_data_frames, labels, columns, 
    output_file=script_dir / "piezo11.xlsx" if zapis_do_souboru else None, 
    save_to_excel=zapis_do_souboru
)

plot_pressure_graphs(final_data_frames, labels, columns, data_s, data_j, orientace, tmin, tmax, script_dir)

# V data frame zapomene všechny listy až na výstup. Tyto listy již nebudou potřeba a jsou obsaženy ve vystup. 
final_data_frames = keep_only_vystup(
    final_data_frames, 
    output_file=script_dir / "piezo12.xlsx" if zapis_do_souboru else None, 
    save_to_excel=zapis_do_souboru
)

# Nalezeni rozdilu mezi radky
# log_large_differences_to_excel_and_csv(script_dir / 'piezo2.xlsx', script_dir / 'output_results.xlsx', script_dir / 'output_results.csv', script_dir / 'rozrazka_nova.xlsx')

final_data_frames = log_large_differences(
    final_data_frames,
    v_diff, a_diff,
    rozrazka_file=script_dir / 'rozrazka_nova.xlsx',
    output_excel_file=script_dir / "output_results.xlsx",
    output_csv_file=script_dir / "output_results.csv"
)
