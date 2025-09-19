"""
Script to clean up global_offshore_dissolve.geojson to match the structure of global_countries_simplify.geojson
"""

import json
import geopandas as gpd
import pandas as pd
from pathlib import Path

def load_geojson(file_path):
    """Load GeoJSON file"""
    print(f"Loading {file_path}...")
    return gpd.read_file(file_path)

def create_country_mapping():
    """Create mapping from COUNTRY names to proper NAME format and GID codes"""
    # Complete mapping for all 89 countries in the offshore data
    country_mapping = {
        'ALGERIA': {'NAME': 'Algeria', 'GID': 'DZA'},
        'ANGOLA': {'NAME': 'Angola', 'GID': 'AGO'},
        'ARGENTINA': {'NAME': 'Argentina', 'GID': 'ARG'},
        'AUSTRALIA': {'NAME': 'Australia', 'GID': 'AUS'},
        'AZERBAIJAN': {'NAME': 'Azerbaijan', 'GID': 'AZE'},
        'BAHRAIN': {'NAME': 'Bahrain', 'GID': 'BHR'},
        'BANGLADESH': {'NAME': 'Bangladesh', 'GID': 'BGD'},
        'BRAZIL': {'NAME': 'Brazil', 'GID': 'BRA'},
        'BRUNEI DARUSSALAM': {'NAME': 'Brunei Darussalam', 'GID': 'BRN'},
        'CAMBODIA': {'NAME': 'Cambodia', 'GID': 'KHM'},
        'CAMEROON': {'NAME': 'Cameroon', 'GID': 'CMR'},
        'CANADA': {'NAME': 'Canada', 'GID': 'CAN'},
        'CHINA': {'NAME': 'China', 'GID': 'CHN'},
        'COLOMBIA': {'NAME': 'Colombia', 'GID': 'COL'},
        'CONGO': {'NAME': 'Congo', 'GID': 'COG'},
        'COTE D\'IVOIRE': {'NAME': 'Côte d\'Ivoire', 'GID': 'CIV'},
        'CUBA': {'NAME': 'Cuba', 'GID': 'CUB'},
        'CYPRUS': {'NAME': 'Cyprus', 'GID': 'CYP'},
        'DEMOCRATIC REPUBLIC OF THE CONGO': {'NAME': 'Democratic Republic of the Congo', 'GID': 'COD'},
        'DENMARK': {'NAME': 'Denmark', 'GID': 'DNK'},
        'DOMINICAN REPUBLIC': {'NAME': 'Dominican Republic', 'GID': 'DOM'},
        'EGYPT': {'NAME': 'Egypt', 'GID': 'EGY'},
        'EL SALVADOR': {'NAME': 'El Salvador', 'GID': 'SLV'},
        'EQUATORIAL GUINEA': {'NAME': 'Equatorial Guinea', 'GID': 'GNQ'},
        'ERITREA': {'NAME': 'Eritrea', 'GID': 'ERI'},
        'ESTONIA': {'NAME': 'Estonia', 'GID': 'EST'},
        'FINLAND': {'NAME': 'Finland', 'GID': 'FIN'},
        'FRANCE': {'NAME': 'France', 'GID': 'FRA'},
        'GABON': {'NAME': 'Gabon', 'GID': 'GAB'},
        'GAMBIA': {'NAME': 'Gambia', 'GID': 'GMB'},
        'GERMANY': {'NAME': 'Germany', 'GID': 'DEU'},
        'GHANA': {'NAME': 'Ghana', 'GID': 'GHA'},
        'GREECE': {'NAME': 'Greece', 'GID': 'GRC'},
        'GUYANA': {'NAME': 'Guyana', 'GID': 'GUY'},
        'INDIA': {'NAME': 'India', 'GID': 'IND'},
        'INDONESIA': {'NAME': 'Indonesia', 'GID': 'IDN'},
        'IRAN': {'NAME': 'Iran', 'GID': 'IRN'},
        'IRELAND': {'NAME': 'Ireland', 'GID': 'IRL'},
        'ISRAEL': {'NAME': 'Israel', 'GID': 'ISR'},
        'ITALY': {'NAME': 'Italy', 'GID': 'ITA'},
        'JAMAICA': {'NAME': 'Jamaica', 'GID': 'JAM'},
        'JAPAN': {'NAME': 'Japan', 'GID': 'JPN'},
        'JORDAN': {'NAME': 'Jordan', 'GID': 'JOR'},
        'KENYA': {'NAME': 'Kenya', 'GID': 'KEN'},
        'LIBYA': {'NAME': 'Libya', 'GID': 'LBY'},
        'LITHUANIA': {'NAME': 'Lithuania', 'GID': 'LTU'},
        'MADAGASCAR': {'NAME': 'Madagascar', 'GID': 'MDG'},
        'MALAYSIA': {'NAME': 'Malaysia', 'GID': 'MYS'},
        'MAURITANIA': {'NAME': 'Mauritania', 'GID': 'MRT'},
        'MEXICO': {'NAME': 'Mexico', 'GID': 'MEX'},
        'MOZAMBIQUE': {'NAME': 'Mozambique', 'GID': 'MOZ'},
        'MYANMAR': {'NAME': 'Myanmar', 'GID': 'MMR'},
        'NETHERLANDS': {'NAME': 'Netherlands', 'GID': 'NLD'},
        'NEW ZEALAND': {'NAME': 'New Zealand', 'GID': 'NZL'},
        'NIGERIA': {'NAME': 'Nigeria', 'GID': 'NGA'},
        'NORWAY': {'NAME': 'Norway', 'GID': 'NOR'},
        'OMAN': {'NAME': 'Oman', 'GID': 'OMN'},
        'PAKISTAN': {'NAME': 'Pakistan', 'GID': 'PAK'},
        'PERU': {'NAME': 'Peru', 'GID': 'PER'},
        'PHILIPPINES': {'NAME': 'Philippines', 'GID': 'PHL'},
        'POLAND': {'NAME': 'Poland', 'GID': 'POL'},
        'QATAR': {'NAME': 'Qatar', 'GID': 'QAT'},
        'REPUBLIC OF KOREA': {'NAME': 'Republic of Korea', 'GID': 'KOR'},
        'ROMANIA': {'NAME': 'Romania', 'GID': 'ROU'},
        'RUSSIAN FEDERATION': {'NAME': 'Russian Federation', 'GID': 'RUS'},
        'SAUDI ARABIA': {'NAME': 'Saudi Arabia', 'GID': 'SAU'},
        'SAUDI ARABIAN–KUWAITI NEUTRAL ZONE': {'NAME': 'Saudi Arabian–Kuwaiti Neutral Zone', 'GID': 'XSA'},
        'SEYCHELLES': {'NAME': 'Seychelles', 'GID': 'SYC'},
        'SINGAPORE': {'NAME': 'Singapore', 'GID': 'SGP'},
        'SOMALIA': {'NAME': 'Somalia', 'GID': 'SOM'},
        'SOUTH AFRICA': {'NAME': 'South Africa', 'GID': 'ZAF'},
        'SPAIN': {'NAME': 'Spain', 'GID': 'ESP'},
        'SWEDEN': {'NAME': 'Sweden', 'GID': 'SWE'},
        'TAIWAN': {'NAME': 'Taiwan', 'GID': 'TWN'},
        'TANZANIA': {'NAME': 'Tanzania', 'GID': 'TZA'},
        'THAILAND': {'NAME': 'Thailand', 'GID': 'THA'},
        'THAILAND MALAYSIA JOINT DEVELOPMENT AREA': {'NAME': 'Thailand Malaysia Joint Development Area', 'GID': 'XTM'},
        'TIMOR-LESTE': {'NAME': 'Timor-Leste', 'GID': 'TLS'},
        'TUNISIA': {'NAME': 'Tunisia', 'GID': 'TUN'},
        'TURKEY': {'NAME': 'Turkey', 'GID': 'TUR'},
        'UKRAINE': {'NAME': 'Ukraine', 'GID': 'UKR'},
        'UNITED ARAB EMIRATES': {'NAME': 'United Arab Emirates', 'GID': 'ARE'},
        'UNITED KINGDOM': {'NAME': 'United Kingdom', 'GID': 'GBR'},
        'UNITED STATES': {'NAME': 'United States', 'GID': 'USA'},
        'VENEZUELA': {'NAME': 'Venezuela', 'GID': 'VEN'},
        'VIETNAM': {'NAME': 'Vietnam', 'GID': 'VNM'},
        'WESTERN SAHARA': {'NAME': 'Western Sahara', 'GID': 'ESH'},
        'YEMEN': {'NAME': 'Yemen', 'GID': 'YEM'},
    }
    return country_mapping

def clean_offshore_data(offshore_gdf, country_mapping):
    """Clean the offshore data to match countries structure"""
    print("Cleaning offshore data...")
    
    # Create a copy to avoid modifying the original
    cleaned_gdf = offshore_gdf.copy()
    
    # Initialize new columns
    cleaned_gdf['NAME'] = None
    cleaned_gdf['GID'] = None
    
    # Map country names and assign GIDs
    for idx, row in cleaned_gdf.iterrows():
        country_name = row['COUNTRY']
        if country_name in country_mapping:
            cleaned_gdf.at[idx, 'NAME'] = country_mapping[country_name]['NAME']
            cleaned_gdf.at[idx, 'GID'] = country_mapping[country_name]['GID']
        else:
            # For unmapped countries, use the original name and create a placeholder GID
            cleaned_gdf.at[idx, 'NAME'] = country_name.title()
            cleaned_gdf.at[idx, 'GID'] = 'UNK'  # Unknown
    
    # Select only the columns that match the countries structure
    cleaned_gdf = cleaned_gdf[['NAME', 'GID', 'geometry']].copy()
    
    # Remove rows where NAME is None (unmapped countries)
    cleaned_gdf = cleaned_gdf.dropna(subset=['NAME'])
    
    print(f"Cleaned data: {len(cleaned_gdf)} features")
    print(f"Unique countries: {cleaned_gdf['NAME'].nunique()}")
    
    return cleaned_gdf

def main():
    """Main function"""
    # Define file paths
    data_dir = Path("data/geojson")
    offshore_file = data_dir / "global_offshore_dissolve.geojson"
    countries_file = data_dir / "global_countries_simplify.geojson"
    output_file = data_dir / "global_offshore_cleaned.geojson"
    
    # Check if files exist
    if not offshore_file.exists():
        print(f"Error: {offshore_file} not found")
        return
    
    if not countries_file.exists():
        print(f"Error: {countries_file} not found")
        return
    
    # Load the offshore data
    offshore_gdf = load_geojson(offshore_file)
    print(f"Original offshore data: {len(offshore_gdf)} features")
    print(f"Original columns: {list(offshore_gdf.columns)}")
    
    # Load countries data to verify structure
    countries_gdf = load_geojson(countries_file)
    print(f"Countries data: {len(countries_gdf)} features")
    print(f"Countries columns: {list(countries_gdf.columns)}")
    
    # Create country mapping
    country_mapping = create_country_mapping()
    
    # Clean the offshore data
    cleaned_gdf = clean_offshore_data(offshore_gdf, country_mapping)
    
    # Save the cleaned data
    print(f"Saving cleaned data to {output_file}...")
    cleaned_gdf.to_file(output_file, driver='GeoJSON')
    
    # Print summary
    print("\nSummary:")
    print(f"Original features: {len(offshore_gdf)}")
    print(f"Cleaned features: {len(cleaned_gdf)}")
    print(f"Unique countries in cleaned data: {cleaned_gdf['NAME'].nunique()}")
    print(f"Output file: {output_file}")
    
    # Show sample of cleaned data
    print("\nSample of cleaned data:")
    print(cleaned_gdf[['NAME', 'GID']].head(10))

if __name__ == "__main__":
    main()
