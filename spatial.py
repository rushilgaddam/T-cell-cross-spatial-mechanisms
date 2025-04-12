import pandas as pd
import numpy as np
import os
import math as ma
import copy
from scipy import stats
import csv

# Constants
base_folder = '/Users/admin/Desktop/NCH data/new_Rushil_Data'
output_folder = '/Users/admin/Desktop/NCH data/'

patient_folder = {
    'f01': os.path.join(base_folder, 'f01'),
    'f02': os.path.join(base_folder, 'f02'),
    'm01': os.path.join(base_folder, 'm01'),
    'm02': os.path.join(base_folder, 'm02')
}

depths = ['Deep', 'Sup']

cell_types = [
    'helper',
    'activated_helper',
    'inhibitory_helper',
    'killer',
    'activated_killer',
    'exhausted_killer',
    'inhibitory_killer'
]

def find_minmax(data):
    if not data:
        return (0, 0, 0, 0)
    
    x_list = [cell[1] for cell in data]  
    y_list = [cell[2] for cell in data]  
    minx = min(x_list)
    maxx = max(x_list)
    miny = min(y_list)
    maxy = max(y_list)

    return minx, maxx, miny, maxy

def get_binned_lattice(data, minmaxL, step, counts_only=False):
    xmin, xmax, ymin, ymax = minmaxL

    latXLen = ma.ceil((xmax - xmin) / step)
    latYLen = ma.ceil((ymax - ymin) / step)

    if latXLen == 0:
        latXLen += 1
    if latYLen == 0:
        latYLen += 1

    if counts_only:
        binnedLat = [[0 for _ in range(latYLen)] for _ in range(latXLen)]
    else:
        binnedLat = [[[] for _ in range(latYLen)] for _ in range(latXLen)]

    for cell in data:
        latX = int((cell[1] - xmin) / step)
        latY = int((cell[2] - ymin) / step)

        if latX == latXLen:
            latX -= 1
        if latY == latYLen:
            latY -= 1

        if 0 <= latX < latXLen and 0 <= latY < latYLen:
            if counts_only:
                binnedLat[latX][latY] += 1
            else:
                toAdd = copy.deepcopy(cell)
                toAdd.append(len(binnedLat[latX][latY]))
                binnedLat[latX][latY].append(toAdd)

    return binnedLat

def find_area(lattice_constant, cell_list, doPrint=False):

    if not cell_list:
        return 0, 0

    minmax = find_minmax(cell_list)
    areaHist = get_binned_lattice(cell_list, minmax, lattice_constant, counts_only=True)

    areaCount = 0
    for row in areaHist:
        for cell_count in row:
            if cell_count > 0:
                areaCount += 1

    area = areaCount * (lattice_constant ** 2)

    if doPrint:
        mmarea = area / 1e6
        print(f"Tissue area = {mmarea:.2f} mm²")
        print(f"With lattice constant: {lattice_constant} µm")

    return area, areaCount

def find_neighbors_in_annulus(iRad, oRad, popList1, popList2, degen=False, symmetric=False):
    neighAv = 0
    neigh2Av = 0

    neighL = [[[0 for _ in range(len(popList1[i][j]))] for j in range(len(popList1[i]))] for i in range(len(popList1))]
    conL = [[[[] for _ in range(len(popList1[i][j]))] for j in range(len(popList1[i]))] for i in range(len(popList1))]

    count = 0
    tCount = 0

    for i in range(len(popList1)):
        for j in range(len(popList1[i])):
            for k in range(len(popList1[i][j])):
                sX = popList1[i][j][k][1]
                sY = popList1[i][j][k][2]

                for l in range(3):
                    for m in range(3):
                        ni = i + l - 1
                        nj = j + m - 1
                        if 0 <= ni < len(popList2) and 0 <= nj < len(popList2[ni]):
                            for n in range(len(popList2[ni][nj])):
                                if popList1[i][j][k][0] != popList2[ni][nj][n][0]:
                                    tX = popList2[ni][nj][n][1]
                                    tY = popList2[ni][nj][n][2]
                                    dist = ma.sqrt((sX - tX) ** 2 + (sY - tY) ** 2)
                                    if iRad <= dist <= oRad:
                                        neighL[i][j][k] += 1
                                        conL[i][j][k].append([
                                            popList1[i][j][k][0],
                                            popList2[ni][nj][n][0],
                                            dist
                                        ])
                                        if degen:
                                            neighL[ni][nj][n] += 1
                                            conL[ni][nj][n].append([
                                                popList2[ni][nj][n][0],
                                                popList1[i][j][k][0],
                                                dist
                                            ])

                if degen:
                    try:
                        popList2[i][j].remove(popList1[i][j][k])
                    except ValueError:
                        pass

                neighAv += neighL[i][j][k]
                neigh2Av += neighL[i][j][k] ** 2
                if neighL[i][j][k] == 0:
                    count += 1
                tCount += 1

    symmetry_factor = 1
    if symmetric:
        symmetry_factor = sum(len(cell) for row in popList2 for cell in row)
        if symmetry_factor == 0:
            symmetry_factor = 1
            print("Warning: Zero target population, outputting asymmetric correlation")

    neighAv /= (tCount * symmetry_factor)
    neigh2Av /= (tCount * symmetry_factor)

    output = {
        'neighbors list': neighL,
        'average': neighAv,
        'second moment': neigh2Av,
        'T cell count': tCount,
        'symmetry': symmetry_factor
    }

    return output

def find_density(popL, area_data, areaLen=30, convertTomm2=False):
    factor = (1e6 - 1) * int(convertTomm2) + 1

    pentrNum = len(popL)
    area, _ = find_area(areaLen, area_data, False)

    peneDens = factor * pentrNum / area if area > 0 else np.nan

    return peneDens, pentrNum

def spatial_correlation(pops, area_data, deltaR=5, start=5, finish=51, areaLen=30, avDensTo0=True, degen=False, scale_by_density=None, symmetric=False, writefile=None, **kwargs):

    distVec = range(start, finish + 1, deltaR)  
    savSlope = np.zeros(len(distVec) - 1)  
    slopes = np.zeros(len(distVec) - 1)  

    slopeX = [start + (i + 0.5) * deltaR for i in range(len(distVec) - 1)] 

    totalPops = pops[0] + pops[1]  
    minmaxL = find_minmax(totalPops)  

    pop2AvDens = find_density(pops[1], area_data, areaLen=areaLen)[0]  

    for p in range(len(distVec) - 1):
        popL1 = get_binned_lattice(pops[0], minmaxL, distVec[p + 1])
        if degen:
            popL2 = copy.deepcopy(popL1)
        else:
            popL2 = get_binned_lattice(pops[1], minmaxL, distVec[p + 1])

        clusterL = find_neighbors_in_annulus(distVec[p], distVec[p + 1], popL1, popL2, degen=degen, symmetric=symmetric)
        savSlope[p] = clusterL['average']

        slopes[p] = (savSlope[p] / (ma.pi * deltaR * (2 * distVec[p] + deltaR))) - (int(avDensTo0) * pop2AvDens)

        if scale_by_density is not None:
            slopes[p] /= scale_by_density

        if writefile is not None:
            lines = [str(distVec[p] + (deltaR / 2)), str(slopes[p]) + '\n']
            writefile.write('\t'.join(lines))

    output = {'radii': slopeX, 'spatial corr vals': slopes}
    return output

def load_data_by_patient(patient_folder, cell_type, depth=None):
    dataframes = []
  
    if 'helper' in cell_type:
        subfolder = 'Helper Combined Files'
    elif 'killer' in cell_type:
        subfolder = 'Killer Combined Files'
    else:
        print(f"Warning: Unknown cell type {cell_type}. Skipping.")
        return []

    for pid, folder in patient_folder.items():
        target_folder = os.path.join(folder, subfolder)
        if not os.path.exists(target_folder):
            print(f"Folder {target_folder} does not exist. Skipping.")
            continue

        expected_filename = f"{pid}_{depth}_{cell_type}.csv"
        file_path = os.path.join(target_folder, expected_filename)

        if not os.path.exists(file_path):
            print(f"Warning: File {expected_filename} not found in {target_folder}. Skipping.")
            continue

        try:
            df = pd.read_csv(file_path)
            if {'Object ID', 'Centroid X µm', 'Centroid Y µm'}.issubset(df.columns):
                cell_data = df[['Object ID', 'Centroid X µm', 'Centroid Y µm']].values.tolist()
                dataframes.extend(cell_data)
            else:
                print(f"Warning: Required columns are missing in {expected_filename}.")
        except Exception as e:
            print(f"Error loading {expected_filename}: {e}")
    return dataframes if dataframes else []

def save_to_csv(data, output_file):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Cell Type A", "Cell Type B", "M01 Deep", "M01 Sup", "M02 Deep", "M02 Sup", "F01 Deep", "F01 Sup", "F02 Deep", "F02 Sup", "P-Value"])
        for row in data:
            writer.writerow(row)

# Main Code Execution 
if __name__ == "__main__":

    cell_type_pairs = [(a, b) for a in cell_types for b in cell_types if a != b]
    results = []

    for cell_type_a, cell_type_b in cell_type_pairs:
        print(f"Processing {cell_type_a} vs {cell_type_b}")

        corr_data_male = []
        corr_data_female = []

        for patient in ['m01', 'm02']:
            for depth in depths:
                data_a = load_data_by_patient({patient: patient_folder[patient]}, cell_type_a, depth)
                data_b = load_data_by_patient({patient: patient_folder[patient]}, cell_type_b, depth)

                if not data_a or not data_b:
                    print(f"Warning: No data found for {patient} {depth} ({cell_type_a} or {cell_type_b}). Skipping.")
                    corr_data_male.append(np.nan)  
                    continue

                combined_data = data_a + data_b

                try:
                    corr_data = spatial_correlation(
                        [data_a, data_b],
                        combined_data,
                        deltaR=5,
                        start=5,
                        finish=51,
                        areaLen=30,
                        avDensTo0=True,
                        degen=False,
                        scale_by_density=None,
                        symmetric=False,
                        writefile=None
                    )

                    corr_vals = corr_data.get('spatial corr vals', [])
                    if len(corr_vals) > 0:
                        corr_data_male.append(float(np.mean(corr_vals)))  
                    else:
                        corr_data_male.append(np.nan)  
                except Exception as e:
                    print(f"Error calculating correlation for {patient} {depth}: {e}")
                    corr_data_male.append(np.nan)  

        for patient in ['f01', 'f02']:
            for depth in depths:
                data_a = load_data_by_patient({patient: patient_folder[patient]}, cell_type_a, depth)
                data_b = load_data_by_patient({patient: patient_folder[patient]}, cell_type_b, depth)

                if not data_a or not data_b:
                    print(f"Warning: No data found for {patient} {depth} ({cell_type_a} or {cell_type_b}). Skipping.")
                    corr_data_female.append(np.nan)  
                    continue

                combined_data = data_a + data_b

                try:
                    corr_data = spatial_correlation(
                        [data_a, data_b],
                        combined_data,
                        deltaR=5,
                        start=5,
                        finish=51,
                        areaLen=30,
                        avDensTo0=True,
                        degen=False,
                        scale_by_density=None,
                        symmetric=False,
                        writefile=None
                    )

                    corr_vals = corr_data.get('spatial corr vals', [])
                    if len(corr_vals) > 0:
                        corr_data_female.append(float(np.mean(corr_vals))) 
                    else:
                        corr_data_female.append(np.nan) 
                except Exception as e:
                    print(f"Error calculating correlation for {patient} {depth}: {e}")
                    corr_data_female.append(np.nan) 

        male_avg = np.nanmean(corr_data_male)
        female_avg = np.nanmean(corr_data_female)
        _, p_value = stats.ttest_ind(corr_data_male, corr_data_female, nan_policy='omit')

        print(f"Correlation data for {cell_type_a} vs {cell_type_b}:")
        print(f"Male: {corr_data_male}")
        print(f"Female: {corr_data_female}")
        print(f"P-value: {p_value}")

        # Store results
        results.append([
            cell_type_a, cell_type_b,
            *corr_data_male,  
            *corr_data_female,  
            p_value  
        ])

    save_to_csv(results, os.path.join(output_folder, "all_cell_type_correlations.csv"))
