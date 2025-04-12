import numpy as np
import math as ma
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy import stats
import csv
from scipy.stats import ttest_ind

def find_minmax(data):
    minx = data['Centroid X µm'].min()
    maxx = data['Centroid X µm'].max()
    miny = data['Centroid Y µm'].min()
    maxy = data['Centroid Y µm'].max()
    if pd.isna(minx) or pd.isna(maxx) or pd.isna(miny) or pd.isna(maxy):
        raise ValueError("NaN values found in centroid coordinates")
    return minx, maxx, miny, maxy

def get_binned_lattice(data, minmaxL, step, counts_only=False):
    xmin, xmax, ymin, ymax = minmaxL
    latXLen = max(1, ma.ceil((xmax - xmin) / step))
    latYLen = max(1, ma.ceil((ymax - ymin) / step))
    binnedLat = [[0 for _ in range(latYLen)] for _ in range(latXLen)] if counts_only else [[list() for _ in range(latYLen)] for _ in range(latXLen)]
    for _, row in data.iterrows():
        latX = min(int((row['Centroid X µm'] - xmin) / step), latXLen - 1)
        latY = min(int((row['Centroid Y µm'] - ymin) / step), latYLen - 1)
        if counts_only: binnedLat[latX][latY] += 1
        else: binnedLat[latX][latY].append(row.to_list())
    return binnedLat

def find_area(lattice_constant, cell_list, doPrint=False):
    minmax = find_minmax(cell_list)
    areaHist = get_binned_lattice(cell_list, minmax, lattice_constant, counts_only=True)
    areaCount = sum(1 for row in areaHist for cell in row if cell > 0)
    area = areaCount * (lattice_constant ** 2)
    if doPrint:
        print(f"tissue area = {area / 1e6:.4f} mm²")
        print(f"with lattice constant: {lattice_constant} µm")
    return area / 1e6, areaCount

def calculate_density_for_cell_type(data, cell_type_conditions, cell_type_name, depth, patient_id, lattice_constant=20, plots_dir='/Users/admin/Desktop/NCH data/out'):
    cell_type_df = data.copy()
    for marker, value in cell_type_conditions.items():
        cell_type_df = cell_type_df[cell_type_df[marker] == value]
    if cell_type_df.empty:
        print("No cells matching the conditions.")
        return 0, 0
    total_area_mm2, total_occupied_cells = find_area(lattice_constant, data, doPrint=True)
    num_cells = len(cell_type_df)
    if num_cells / total_area_mm2 < 1: lattice_constant = max(10, lattice_constant)
    print(f"Number of cells matching conditions {cell_type_conditions}: {num_cells}")
    print(f"Occupied lattice cells (Total Tissue Area): {total_occupied_cells}")
    if total_area_mm2 == 0:
        print("Total tissue area is zero. Cannot compute density.")
        return 0, 0
    density = num_cells / total_area_mm2
    print(f"Calculated Density: {density:.2f} cells/mm²")
    return density, total_area_mm2

def get_patient_density(patient_id, depth, cell_type_name, cell_type_conditions=None, lattice_constant=20, plots_dir='/Users/admin/Desktop/NCH data/out'):
    depth_str = 'Deep' if depth.lower() == 'deep' else 'Sup'
    path = f'/Users/admin/Desktop/NCH data/new_Rushil_Data/{patient_id}/Regular Combined Files/{patient_id}_{depth_str}_combined.csv'
    if not os.path.exists(path): raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    combined_df = df.dropna(subset=['Centroid X µm', 'Centroid Y µm'])
    if cell_type_conditions:
        return calculate_density_for_cell_type(combined_df, cell_type_conditions, cell_type_name, depth, patient_id, lattice_constant=lattice_constant, plots_dir=plots_dir)
    else: raise ValueError("cell_type_conditions must be provided")

def save_densities_to_csv(densities_female, densities_male, depth, output_dir):
    combined_densities = {'Patient_ID': list(densities_female.keys()) + list(densities_male.keys()),
                         'Density': list(densities_female.values()) + list(densities_male.values()),
                         'Gender': ['Female'] * len(densities_female) + ['Male'] * len(densities_male),
                         'Depth': [depth] * (len(densities_female) + len(densities_male))}
    density_df = pd.DataFrame(combined_densities)
    file_path = os.path.join(output_dir, f"{depth}_densities.csv")
    density_df.to_csv(file_path, index=False)
    print(f"Densities saved to {file_path}")

def plot_density(depth, cell_type_name, avg_female, avg_male, densities_female, densities_male, plots_dir):
    categories = ['Female', 'Male']
    averages = [avg_female, avg_male]
    x = np.arange(len(categories))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, averages, 0.6, color=['deeppink', 'darkblue'], alpha=0.6, label='_nolegend_')
    handled_labels = set()
    for i, category in enumerate(categories):
        if category == 'Female':
            patient_ids = list(densities_female.keys())
            patient_densities = list(densities_female.values())
            dot_colors = ['pink', 'lightcoral']
        else:
            patient_ids = list(densities_male.keys())
            patient_densities = list(densities_male.values())
            dot_colors = ['lightblue', 'royalblue']
        for j, (pid, density) in enumerate(zip(patient_ids, patient_densities)):
            label = pid if pid not in handled_labels else "_nolegend_"
            ax.scatter(x[i], density, color=dot_colors[j % len(dot_colors)], zorder=5, label=label, s=100)
            handled_labels.add(pid)
    ax.set_xlabel('Gender', fontsize=14)
    ax.set_ylabel('Cell Density (cells/mm²)', fontsize=14)
    ax.set_title(f'Cell Density of {cell_type_name} ({depth})', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys(), title='Patients', loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{cell_type_name.lower().replace(" ", "_")}_density_{depth.lower()}.png'), dpi=300)
    plt.close()

def calculate_p_values(density_data):
    df = pd.DataFrame(density_data, columns=['Group', 'Patient_ID', 'Depth', 'Cell_Type', 'Density_cells_per_mm2'])
    p_values = {}
    for cell_type in df['Cell_Type'].unique():
        cell_type_data = df[(df['Cell_Type'] == cell_type) & (df['Density_cells_per_mm2'] > 0)]
        female_densities = cell_type_data[cell_type_data['Group'] == 'Female']['Density_cells_per_mm2']
        male_densities = cell_type_data[cell_type_data['Group'] == 'Male']['Density_cells_per_mm2']
        male_female_p_value = ttest_ind(female_densities, male_densities).pvalue if len(female_densities) > 1 and len(male_densities) > 1 else np.nan
        deep_densities = cell_type_data[cell_type_data['Depth'] == 'Deep']['Density_cells_per_mm2']
        superficial_densities = cell_type_data[cell_type_data['Depth'] == 'Superficial']['Density_cells_per_mm2']
        deep_sup_p_value = ttest_ind(deep_densities, superficial_densities).pvalue if len(deep_densities) > 1 and len(superficial_densities) > 1 else np.nan
        p_values[cell_type] = {'Male_vs_Female_p_value': male_female_p_value, 'Deep_vs_Superficial_p_value': deep_sup_p_value,
                              'Male_vs_Female_data': {'Female': female_densities.tolist(), 'Male': male_densities.tolist()},
                              'Deep_vs_Superficial_data': {'Deep': deep_densities.tolist(), 'Superficial': superficial_densities.tolist()}}
    return p_values

def save_p_values_to_csv(p_values, output_dir):
    rows = []
    for cell_type, values in p_values.items():
        rows.append({'Cell_Type': cell_type, 'Male_vs_Female_p_value': values['Male_vs_Female_p_value'],
                    'Deep_vs_Superficial_p_value': values['Deep_vs_Superficial_p_value'],
                    'Male_vs_Female_data_Female': values['Male_vs_Female_data']['Female'],
                    'Male_vs_Female_data_Male': values['Male_vs_Female_data']['Male'],
                    'Deep_vs_Superficial_data_Deep': values['Deep_vs_Superficial_data']['Deep'],
                    'Deep_vs_Superficial_data_Superficial': values['Deep_vs_Superficial_data']['Superficial']})
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, 'p_values_with_data.csv'), index=False)
    print(f"P-values and data points saved to {os.path.join(output_dir, 'p_values_with_data.csv')}")

def main():
    cell_types = {
        'Killer CD8+ T-cells': {'CD3': 1, 'CD8': 1},
        'Activated Killer T-cells': {'CD3': 1, 'CD8': 1, 'Tim3': 0, 'PD-1': 1},
        'Exhausted Killer T-cells': {'CD3': 1, 'CD8': 1, 'Tim3': 1},
        'Inhibitory Killer T-cells': {'CD3': 1, 'CD8': 1, 'Tim3': 1, 'ICOS': 1},
        'Helper T-cells': {'CD3': 1, 'CD4': 1},
        'Activated Helper T-cells': {'CD3': 1, 'CD4': 1, 'PD-1': 1},
        'Inhibitory Helper T-cells': {'CD3': 1, 'CD4': 1, 'ICOS': 1}
    }
    
    female_patients = ['f01', 'f02']
    male_patients = ['m01', 'm02']
    plots_dir = '/Users/admin/Desktop/NCH data/out'
    os.makedirs(plots_dir, exist_ok=True)
    
    lattice_constant = 20
    density_data = []
    slide_areas = []

    for cell_type, conditions in cell_types.items():
        for depth in ['Deep', 'Sup']:
            densities_female = {}
            densities_male = {}
            
            for pid in female_patients:
                try:
                    density, area = get_patient_density(pid, depth, cell_type, conditions, lattice_constant, plots_dir)
                    densities_female[pid] = density
                    density_data.append(['Female', pid, depth, cell_type, density])
                    slide_areas.append({'Patient_ID': pid, 'Depth': depth, 'Cell_Type': cell_type, 'Area_mm2': area})
                except Exception as e: print(f"Error {pid} {depth}: {e}")
            
            for pid in male_patients:
                try:
                    density, area = get_patient_density(pid, depth, cell_type, conditions, lattice_constant, plots_dir)
                    densities_male[pid] = density
                    density_data.append(['Male', pid, depth, cell_type, density])
                    slide_areas.append({'Patient_ID': pid, 'Depth': depth, 'Cell_Type': cell_type, 'Area_mm2': area})
                except Exception as e: print(f"Error {pid} {depth}: {e}")
            
            avg_female = np.mean(list(densities_female.values())) if densities_female else 0
            avg_male = np.mean(list(densities_male.values())) if densities_male else 0
            
            plot_density(depth, cell_type, avg_female, avg_male, densities_female, densities_male, plots_dir)
    
    pd.DataFrame(density_data, columns=['Group','Patient_ID','Depth','Cell_Type','Density']).to_csv(
        os.path.join(plots_dir, 'cell_density_data.csv'), index=False)
    
    p_values = calculate_p_values(density_data)
    save_p_values_to_csv(p_values, plots_dir)
    
    pd.DataFrame(slide_areas).to_csv(os.path.join(plots_dir, 'slide_areas.csv'), index=False)

if __name__ == "__main__":
    main()

  
