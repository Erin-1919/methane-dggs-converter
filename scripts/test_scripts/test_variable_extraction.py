#!/usr/bin/env python3
"""
Test script to verify the variable extraction fix works correctly
"""

from global_gfei_netcdf_to_dggs_optimize import GlobalGFEINetCDFToDGGSConverterOptimized

def test_variable_extraction():
    # Test the fixed extract_variable_from_filename method
    test_files = [
        'Global_Fuel_Exploitation_Inventory_Gas_All.nc',
        'Global_Fuel_Exploitation_Inventory_v2_2019_Gas_All.nc', 
        'Global_Fuel_Exploitation_Inventory_v3_2020_Gas_All.nc',
        'Global_Fuel_Exploitation_Inventory_Gas_Exploration.nc',
        'Global_Fuel_Exploitation_Inventory_Gas_Distribution.nc'
    ]

    print('Testing variable extraction:')
    print('=' * 50)
    for filename in test_files:
        variable = GlobalGFEINetCDFToDGGSConverterOptimized.extract_variable_from_filename(filename)
        print(f'{filename}')
        print(f'  -> Extracted variable: {variable}')
        print()

if __name__ == "__main__":
    test_variable_extraction()
