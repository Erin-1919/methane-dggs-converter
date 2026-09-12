# -*- coding: utf-8 -*-
"""Generate the complete original-variable to IPCC 2006 sector crosswalk.

The crosswalk is derived from the per-inventory lookup tables the converters load, so it
documents the mapping the code actually applied rather than a hand-maintained copy of it.
One row per original variable per converted dataset, recording the cardinality of the
mapping and the basis on which the IPCC 2006 category was assigned.

Cardinality:
  1:1       one variable, one IPCC 2006 code, and no other variable in that inventory
            carries the same code
  many:1    several variables in the inventory share one code; the converter sums them
  1:many    the variable spans several IPCC 2006 categories that the source does not
            separate, recorded as a '+'-joined label and carried through as one column
  retained  no IPCC 2006 category applies, so the original label is kept (the Swiss
            natural and semi-natural fluxes)

Basis:
  native    the source carries IPCC 2006 codes, adopted directly
  crosswalk the source carries CRT, GNFR/CRF or IPCC 1996 codes, mapped through the
            documented correspondence tables in data/lookup/
  judgement the source provides descriptive variable names only, and the category was
            assigned by reading the source publication and the IPCC category definitions

Run from the repo root:
    python scripts/analysis/build_sector_crosswalk.py [--write]
"""
import io
import os
import sys

import pandas as pd

ROOT = 'E:/UCalgary_postdoc/genAI_dggs_ch4/methane_grid_calculation_ARC/'
LOOKUP = ROOT + 'data/lookup/'
OUT = LOOKUP + 'ipcc2006_sector_crosswalk_complete.csv'
SUMMARY = LOOKUP + 'ipcc2006_sector_crosswalk_summary.csv'

# Two lookup files hold more than one dataset. Their variables are converted separately,
# one output per country, so the cardinality must be counted per dataset and not per file.
SPLITS = {
    'cms_netcdf_variable_lookup.csv': [
        ('CMS Canada', lambda v: '_CA_' in v),
        ('CMS Mexico', lambda v: '_MX_' in v),
    ],
    'india_australia_csv_variable_lookup.csv': [
        ('Australia coal mine', lambda v: '_aus_' in v),
        ('India coal mine', lambda v: '_ind_' in v),
    ],
}

# file, inventory label, variable column, native-code column, native scheme, basis
SOURCES = [
    ('edgar_ipcc_sector_mapping.csv', 'EDGAR v8.0',
     'EDGAR Sector', 'IPCC1996', 'EDGAR sector abbreviations with IPCC 1996 codes', 'crosswalk'),
    ('gfei_netcdf_variable_lookup.csv', 'GFEI',
     'variable', None, 'Descriptive variable names', 'judgement'),
    ('cams-reg_netcdf_variable_lookup.csv', 'CAMS-REG v8.1',
     'variable', 'CRF', 'GNFR14 sectors with CRF codes', 'crosswalk'),
    ('us_netcdf_variable_lookup.csv', 'U.S. Anthropogenic (Gridded GHGI)',
     'variable', 'CRT', 'CRT codes embedded in variable names', 'crosswalk'),
    ('us_OG_netcdf_variable_lookup.csv', 'U.S. Oil and Gas',
     'variable', None, 'Single unlabelled emission field', 'judgement'),
    ('canada_netcdf_variable_lookup.csv', 'Canada',
     'variable', 'CRT', 'CRT codes embedded in variable names', 'crosswalk'),
    ('mexico_netcdf_variable_lookup.csv', 'Mexico',
     'variable', 'CRT', 'CRT codes embedded in variable names', 'crosswalk'),
    ('cms_netcdf_variable_lookup.csv', 'CMS Canada and Mexico',
     'variable', None, 'Descriptive variable names', 'judgement'),
    ('china_tiff_variable_lookup.csv', 'China CHN-CH4',
     'variable', None, 'Descriptive variable names', 'judgement'),
    ('china_SACMS_netcdf_variable_lookup.csv', 'China coal mine (SACMS)',
     'variable', None, 'Single unlabelled emission field', 'judgement'),
    ('swiss_netcdf_variable_lookup.csv', 'Switzerland',
     'variable', None, 'Descriptive variable names', 'judgement'),
    ('gnys_netcdf_variable_lookup.csv', 'New York State',
     'variable', 'IPCC1996', 'IPCC 1996 codes embedded in variable names', 'crosswalk'),
    ('india_australia_csv_variable_lookup.csv', 'India and Australia coal mine',
     'variable', None, 'Descriptive variable names', 'judgement'),
]

# The Swiss product spells its landfill variable 'lanfill'. The lookup key has to keep
# that spelling or the converter skips the variable and the 4A column disappears, so the
# correction is applied to the published name only.
RENAMED = {('Switzerland', 'lanfill'): 'landfill'}

# Judgement calls and structural cases that need a note of their own. Keyed by
# (inventory, variable); anything not listed takes the automatic note.
NOTES = {
    ('Switzerland', 'forest'): 'Natural flux, retained under the original label.',
    ('Switzerland', 'lake'): 'Natural flux, retained under the original label.',
    ('Switzerland', 'wetland'): 'Natural flux, retained under the original label.',
    ('Switzerland', 'wild_animal'): 'Natural flux, retained under the original label.',
    ('CAMS-REG v8.1', 'J_Waste'):
        'CRF sector 5 corresponds to IPCC 2006 sector 4, following the renumbering of the '
        'waste sector between the 1996 and 2006 guidelines.',
}


def cardinality(assigned, variable, counts):
    if assigned == variable:
        return 'retained'
    if '+' in str(assigned):
        return '1:many'
    if counts.get(assigned, 0) > 1:
        return 'many:1'
    return '1:1'


def auto_note(kind, assigned, basis, scheme):
    """A note only where it adds something the other columns do not already say."""
    if kind == '1:many':
        return 'Source does not separate the %d categories.' % len(str(assigned).split('+'))
    if kind == 'many:1':
        return 'Summed with the other variables sharing this code.'
    return ''


def dataset_parts(fname, inventory, vcol, frame):
    """One (label, frame) per converted dataset, splitting the multi-country files."""
    if fname not in SPLITS:
        return [(inventory, frame)]
    return [(label, frame[frame[vcol].astype(str).map(pick)])
            for label, pick in SPLITS[fname]]


def build():
    rows = []
    for fname, inventory, vcol, ncol, scheme, basis in SOURCES:
        full = pd.read_csv(LOOKUP + fname)
        for label, df in dataset_parts(fname, inventory, vcol, full):
            counts = df['IPCC2006'].value_counts().to_dict()
            for _, r in df.iterrows():
                variable = str(r[vcol]).strip()
                assigned = str(r['IPCC2006']).strip()
                kind = cardinality(assigned, variable, counts)
                native = '' if ncol is None else str(r[ncol]).strip()
                row_basis = 'retained' if kind == 'retained' else basis
                note = NOTES.get((label, variable),
                                 auto_note(kind, assigned, row_basis, scheme))
                rows.append({
                    'Inventory': label,
                    'Native coding scheme': scheme,
                    'Original variable': RENAMED.get((label, variable), variable),
                    'Native code': '' if native == 'nan' else native,
                    'IPCC 2006 code': '' if kind == 'retained' else assigned,
                    'Mapping': kind,
                    'Basis': row_basis,
                    'Note': note,
                })
    return pd.DataFrame(rows)


def summarize(tab):
    out = []
    for inv, g in tab.groupby('Inventory', sort=False):
        counts = g['Mapping'].value_counts()
        out.append({
            'Inventory': inv,
            'Native coding scheme': g['Native coding scheme'].iloc[0],
            'Variables': len(g),
            '1:1': int(counts.get('1:1', 0)),
            'many:1': int(counts.get('many:1', 0)),
            '1:many': int(counts.get('1:many', 0)),
            'retained': int(counts.get('retained', 0)),
            'Judgement': int((g['Basis'] == 'judgement').sum()),
            'Share 1:1 (%)': round(100.0 * counts.get('1:1', 0) / len(g), 1),
        })
    total = tab['Mapping'].value_counts()
    out.append({
        'Inventory': 'All inventories',
        'Native coding scheme': '',
        'Variables': len(tab),
        '1:1': int(total.get('1:1', 0)),
        'many:1': int(total.get('many:1', 0)),
        '1:many': int(total.get('1:many', 0)),
        'retained': int(total.get('retained', 0)),
        'Judgement': int((tab['Basis'] == 'judgement').sum()),
        'Share 1:1 (%)': round(100.0 * total.get('1:1', 0) / len(tab), 1),
    })
    return pd.DataFrame(out)


if __name__ == '__main__':
    table = build()
    summary = summarize(table)
    print(summary.to_string(index=False))
    print('\ntotal rows: %d' % len(table))
    if '--write' in sys.argv:
        table.to_csv(OUT, index=False, encoding='utf-8-sig')
        summary.to_csv(SUMMARY, index=False, encoding='utf-8-sig')
        print('\nwritten:\n  %s\n  %s' % (OUT, SUMMARY))
    else:
        print('\ndry run - pass --write to save the CSVs')
