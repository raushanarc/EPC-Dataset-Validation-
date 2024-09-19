__Author__ = "Kumar Raushan"
__email__ = "d19125689@mytudublin.ie"
__Reference work__ "K. Raushan, T. Mac Uidhir, M. Llorens Salvador, B. Norton, C. Ahern, A data-driven standardised generalisable methodology to validate a large energy performance Certification dataset: A case of the application in Ireland, Energy and Buildings (2024) 114774. https://doi.org/10.1016/j.enbuild.2024.114774. "

"""
This Module Contains all Functions required form the Data Cleaning Process
- Add Desired Columns
- Drop Unwanted Columns
- Find and Replace
- Filter by Criteria
"""

import pandas as pd
import numpy as np
from datetime import datetime


# ------- Lookup Dictionaries ----------
# Lightweight data stored in global space to limit uneccessary 
# duplication when utilized with multiprocessing and threading.

# Keys [ dwelling_type, feature ] -> (min, max)
typology_inclusion_dict = {
    'Semi-detached house': {
        'LivingAreaPercent': (11.27, 35.16),
        'WallArea': (53.19, 147.96),
        'FloorArea': (31.13, 119.09),
        'GroundFloorArea(sq m)': (48.75, 186.83),
        'RoofArea': (32.34, 127.39),
        'WindowArea': (7.09, 41.99),
        'DoorArea': (0, 5.86),
    },
    'End of terrace house': {
        'LivingAreaPercent': (12.96, 59.49),
        'WallArea': (56.62, 155.61),
        'FloorArea': (29.97, 104.74),
        'GroundFloorArea(sq m)': (58.68, 190.49),
        'RoofArea': (29.99, 108.82),
        'WindowArea': (5.66, 34.82),
        'DoorArea': (1.54, 17.97),
    },
    'Detached house': {
        'LivingAreaPercent': (7.54, 45.89),
        'WallArea': (31.68, 336.19),
        'FloorArea': (28.87, 255.60),
        'GroundFloorArea(sq m)': (52.24, 422.20),
        'RoofArea': (30.08, 290.85),
        'WindowArea': (5.61, 83.48),
        'DoorArea': (0, 6.28),
    },
    'Top-floor apartment': {
        'LivingAreaPercent': (-0.23, 62.28),
        'WallArea': (4.63, 139.35),
        'FloorArea': (0, 0),
        'GroundFloorArea(sq m)': (18.29, 153.06),
        'RoofArea': (12.16, 126.58),
        'WindowArea': (2.02, 38.71),
        'DoorArea': (0, 1.91),
    },
    'Mid-terrace house': {
        'LivingAreaPercent': (13.03, 72.13),
        'WallArea': (29.24, 163.64),
        'FloorArea': (31.65, 101.19),
        'GroundFloorArea(sq m)': (61.13, 189.89),
        'RoofArea': (32.54, 110.68),
        'WindowArea': (4.29, 29.02),
        'DoorArea': (1.61, 21.04),
    },
    'Maisonette': {
        'LivingAreaPercent': (6.39, 78.28),
        'WallArea': (20.88, 255.43),
        'FloorArea': (-3.65, 841.0),
        'GroundFloorArea(sq m)': (12.82, 182.30),
        'RoofArea': (-121.43, 84.78),
        'WindowArea': (1.83, 35.75),
        'DoorArea': (1.83, 6.21),
    },
    'House': {
        'LivingAreaPercent': (6.61, 39.73),
        'WallArea': (57.96, 392.64),
        'FloorArea': (-8.56, 248.01),
        'GroundFloorArea(sq m)': (53.63, 453.75),
        'RoofArea': (-4.44, 284.29),
        'WindowArea': (5.44, 78.07),
        'DoorArea': (0, 4.73),
    },
    'Apartment': {
        'LivingAreaPercent': (43.39, 57.59),
        'WallArea': (0.03, 135.15),
        'FloorArea': (-1.41, 1596.88),
        'GroundFloorArea(sq m)': (14.73, 143.84),
        'RoofArea': (-4.74, 725.23),
        'WindowArea': (1.72, 47.62),
        'DoorArea': (1.78, 2.02),
    },
    'Ground-floor apartment': {
        'LivingAreaPercent': (3.72, 64.30),
        'WallArea': (-5.79, 113.86),
        'FloorArea': (2.23, 107.71),
        'GroundFloorArea(sq m)': (14.74, 125.67),
        'RoofArea': (-0.19, 214.01),
        'WindowArea': (2.96, 32.76),
        'DoorArea': (1.45, 2.40),
    },
    'Mid-floor apartment': {
        'LivingAreaPercent': (21.59, 65.04),
        'WallArea': (-0.75, 104.89),
        'FloorArea': (0, 0),
        'GroundFloorArea(sq m)': (8.61, 114.22),
        'RoofArea': (0, 0),
        'WindowArea': (2.77, 41.77),
        'DoorArea': (0, 1.91),
    },
    'Basement Dwellinge': {
        'LivingAreaPercent': (-4.70, 81.83),
        'WallArea': (0.94, 148.01),
        'FloorArea': (-12.76, 141.22),
        'GroundFloorArea(sq m)': (2.43, 189.37),
        'RoofArea': (-0.21, 238.13),
        'WindowArea': (-0.28, 32.29),
        'DoorArea': (0.29, 2.23),
    },
}

# Keys [ rating, (uvalue, walltype) ] -> uvalue
uvalue_wall_replacement_dict = {
    'A': {(2.10, 'Unknown'): 2.69, (2.10, 'Stone'): 2.90, (1.64, '325mm Solid Brick'): 1.55},
    'B': {(2.10, 'Unknown'): 1.75, (2.10, 'Stone'): 3.28, (2.10, '225mm Solid Brick'): 1.75, (1.64, '325mm Solid Brick'): 1.55},
    'C': {(2.10, 'Unknown'): 2.12, (2.10, '225mm Solid brick'): 1.75, (1.78, '300mm Cavity'): 1.20, (2.20, 'Solid Mass Concrete'): 2.12},
    'D': {(2.10, 'Unknown'): 2.69, (1.78, '300mm Cavity'): 1.85, (2.20, 'Solid Mass Concrete'): 2.12, (2.40, 'Concrete Hollow Block'): 2.69},
    'E': {(2.10, 'Unknown'): 2.14, (1.78, '300mm Cavity'): 1.54, (2.40, 'Concrete Hollow Block'): 2.14},
    'F': {(1.10, 'Unknown'): 1.83, (1.10, '300mm Cavity'): 1.43, (0.60, '300mm Filled Cavity'): 0.54, (1.10, 'Concrete Hollow Block'): 1.83},
    'G': {(0.60, 'Unknown'): 1.35, (0.60, '300mm Cavity'): 1.35, (0.60, '300mm Filled Cavity'): 0.54, (0.60, 'Concrete Hollow Block'): 1.72},
    'H': {(0.55, 'Unknown'): 0.39, (0.55, '300mm Filled Cavity'): 0.39, (0.55, 'Concrete Hollow Block'): 0.54, (0.55, 'Timber Frame'): 0.40},
    'I': {(0.55, 'Unknown'): 0.28, (0.55, '300mm Filled Cavity'): 0.29, (0.55, 'Timber Frame'): 0.35},
    'J': {(0.37, 'Unknown'): 0.27, (0.37, '300mm Filled Cavity'): 0.27, (0.37, 'Timber Frame'): 0.30},
    'K': {(0.27, 'Unknown'): 0.27, (0.27, '300mm Filled Cavity'): 0.21, (0.27, 'Timber Frame'): 0.27},
}

# Keys [ rating, uvalue ] -> uvalue
uvalue_roof_replacement_dict = {
    'A': {2.30: 0.71},
    'B': {2.30: 0.71},
    'C': {2.30: 0.71},
    'D': {2.30: 0.71},
    'E': {2.30: 0.71},
    'F': {0.49: 0.43},
    'G': {0.49: 0.43},
    'H': {0.40: 0.28},
    'I': {0.36: 0.28},
    'J': {0.25: 0.21},
    'K': {0.25: 0.21},
}

# -------- Dictionary Lookup Functions ----------
def typology_filter_function(data_row:pd.Series) -> bool:
    criteria = typology_inclusion_dict.get(data_row['DwellingTypeDescr'], {})
    for key, (min, max) in criteria.items():
        if not (min <= data_row[key] <= max):
            return False
    return True

def uv_wall_replacement_function(data_row:pd.Series) -> pd.Series:
    data_row['UValueWall'] = uvalue_wall_replacement_dict[data_row['AgeBand']].get((data_row['UValueWall'], data_row['FirstWallType_Description']), data_row['UValueWall'])
    return data_row

def uv_roof_replacement_function(data_row:pd.Series) -> pd.Series:
    data_row['UValueRoof'] = uvalue_roof_replacement_dict[data_row['AgeBand']].get(data_row['UValueRoof'], data_row['UValueRoof'])
    return data_row



# -------- Main Cleaning Functions ------------
def add_desired_columns(data:pd.DataFrame) -> pd.DataFrame:

    bins = [float('-inf'), 1899, 1929, 1949, 1966, 1977, 1982, 1993, 1999, 2004, 2009, float('inf')]
    construction_period = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]
    data['AgeBand'] = pd.cut(data['Year_of_Construction'], bins=bins, labels=construction_period, right=True)

    data['ThermalEra'] = data['Year_of_Construction'].apply(lambda x: 'Pre' if x <= 1977 else 'Post')

    data['GlazingPercent'] = data['WindowArea'] / data['WallArea'].replace(0, np.nan)
    data['GlazingPercent'] = data['GlazingPercent'].replace(np.nan, 0)

    data['Volume'] = data['GroundFloorArea(sq m)'] * (data['GroundFloorHeight'] + data['FirstFloorHeight'] + data['SecondFloorHeight'] + data['ThirdFloorHeight'])
    
    data['Location'] = data['CountyName'].apply(lambda x: 'urban' if any (criteria in x for criteria in ['City', 'Dublin ']) else 'rural' if 'Co.' in x else 'unknown')

    return data


def drop_unwanted_columns(data:pd.DataFrame) -> pd.DataFrame:
    # There are errors within the description columns due to thier subjective nature
    # so it was decided that they be dropped
    columns_to_drop  = ['FirstWallDescription', 'SecondWallDescription', 'ThirdWallDescription']
    columns_to_drop += [col for col in data.columns if col.startswith('Unnamed:')]
    kept_columns     = [col for col in data.columns if col not in columns_to_drop]

    return data[kept_columns]


def find_and_replace(data:pd.DataFrame) -> pd.DataFrame:
    # Correct the County Name
    counties_list = ['Carlow', 'Cavan', 'Clare', 'Cork', 'Donegal', 'Dublin', 'Galway', 'Kerry', 'Kildare', 'Kilkenny', 'Laois', 'Leitrim', 'Limerick', 'Longford', 'Louth', 'Mayo', 'Meath', 'Monaghan', 'Offaly', 'Roscommon', 'Sligo', 'Tipperary', 'Waterford', 'Westmeath', 'Wexford', 'Wicklow']

    data.loc[:, 'CountyName'] = data['CountyName']\
        .apply(lambda x: next((county for county in counties_list if county in x), x))\
        .str.strip()
    
    # Correctly assign number of stories. Staring from 1 storey for a solo ground floor building.
    height_columns = ['FirstFloorHeight', 'SecondFloorHeight', 'ThirdFloorHeight']
    data.loc[:, 'NoStoreys'] = data.apply(
        lambda row: (1 + sum(row[height_columns].gt(0))) if (row['NoStoreys'] < 4) else row['NoStoreys'], axis=1
    )
    
    data.loc[:, 'EnergyRating'] = data['EnergyRating'].astype(str).str.strip()

    data = data\
        .apply(uv_wall_replacement_function, axis=1)\
        .apply(uv_roof_replacement_function, axis=1)

    return data


def filter_by_criteria(data:pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # -------- Filter conditions ----------
    conditions_thermal_pre = (
        (data['ThermalEra'] == 'Pre')    &
        (((1.10 <= data['UvalueDoor'])   & (data['UvalueDoor']   <= 3.90)) | (data['UvalueDoor'] == 0))   &
        (((1.18 <= data['UValueWindow']) & (data['UValueWindow'] <= 5.70)) | (data['UValueWindow'] == 0)) &
        (((0.13 <= data['UValueRoof'])   & (data['UValueRoof']   <= 1.99)) | (data['UValueRoof'] == 0))   &
        (((0.16 <= data['UValueFloor'])  & (data['UValueFloor']  <= 1.14)) | (data['UValueFloor'] == 0))  &
         (0.20 <= data['UValueWall'])   & (data['UValueWall']   <= 2.90) # not allowed to be zero
    )
    
    conditions_thermal_post = (
        (data['ThermalEra'] == 'Post')  &
        (((0.83 <= data['UvalueDoor'])   & (data['UvalueDoor']   <= 3.54)) | (data['UvalueDoor'] == 0))   &
        (((0.77 <= data['UValueWindow']) & (data['UValueWindow'] <= 4.80)) | (data['UValueWindow'] == 0)) &
        (((0.11 <= data['UValueRoof'])   & (data['UValueRoof']   <= 0.68)) | (data['UValueRoof'] == 0))   &
        (((0.11 <= data['UValueFloor'])  & (data['UValueFloor']  <= 1.14)) | (data['UValueFloor'] == 0))  &
         (0.14 <= data['UValueWall'])   & (data['UValueWall']   <= 1.72) # not allowed to be zero
    )
    
    conditions_general = (
        (data['Year_of_Construction'] <= datetime.today().year) &

        (((2.30 <= data['GroundFloorHeight']) & (data['GroundFloorHeight'] <= 3.46)) |
        (data['GroundFloorHeight'] == 0)) &

        (data['TypeofRating'].str.contains('Final|Existing|Provisional')) &
   
        (((23.07 <= data['HSMainSystemEfficiency']) & (data['HSMainSystemEfficiency']  <= 95.90)) |   # conventional boiler
        ((100   <= data['HSMainSystemEfficiency']) & (data['HSMainSystemEfficiency']  <= 635.34)) | # prob. central heating system
        (data['HSMainSystemEfficiency'] == 0)) &   # no central heating system

        (((24  <= data['WHMainSystemEff']) & (data['WHMainSystemEff']  <= 95.90)) |    # conventional boiler
        ((100  <= data['WHMainSystemEff']) & (data['WHMainSystemEff']  <= 389.9)) |   # prob. heat pumps
        (data['WHMainSystemEff'] == 0))     # no central heating system
    )

    # Composing the filter mask
    thermal_conditions = conditions_thermal_pre | conditions_thermal_post
    general_conditions = conditions_general & data.apply(typology_filter_function, axis=1)
    filter_mask = thermal_conditions & general_conditions

    return data[filter_mask], data[~filter_mask]



# ----- Main function ------
def process_data(data: pd.DataFrame, filter_flag: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = add_desired_columns(data)
    data = drop_unwanted_columns(data)
    data = find_and_replace(data)
    exclustion = None

    if filter_flag:
        data, exclustion = filter_by_criteria(data)

    return data, exclustion


# run unit tests here
if __name__ == '__main__':
    input_file_path = '../Data/BER Snapshots/Full Unfiltered/2020 (Q2) April.csv'
    output_file_path = '../Data/BER Snapshots/temp.csv'
    excluded_output_file_path = '.'.join(output_file_path.split('.')[:-1]) + '_excluded.csv'

    first_write = True
    with open(output_file_path, 'w', encoding='utf-8', errors='ignore') as output_file, open(excluded_output_file_path, 'w', encoding='utf-8', errors='ignore') as excluded_output_file:
        for chunk in pd.read_csv(input_file_path, chunksize=10000, low_memory=False, encoding='utf-8'):
            res, fil = process_data(chunk, filter_flag=True)

            res.to_csv(output_file, header=first_write)
            fil.to_csv(excluded_output_file, header=first_write)
            first_write = False
    

    # result_data = pd.DataFrame()
    # excluded_data = pd.DataFrame()
    # first_write = True

    # for chunk in pd.read_csv(input_file_path, chunksize=10000, low_memory=False, encoding='utf-8'):
    #     res, ex = process_data(chunk, exter_flag=True)
    #     if first_write:
    #         result_data, excluded_data = res, ex
    #     else:
    #         result_data = pd.concat([result_data, res])
    #         excluded_data = pd.concat([excluded_data, ex])
    #     break

    # result_data.to_csv(output_file_path, index=False, header=True, encoding='utf-8', errors='ignore')
    # excluded_data.to_csv(excluded_output_file_path, index=False, header=True, encoding='utf-8', errors='ignore')
    
    # print(f'Data Length - {len(result_data)}, Chunk Length - {10000}')
    # print(result_data.head())
    # print(result_data.describe())
    # print(result_data.info())