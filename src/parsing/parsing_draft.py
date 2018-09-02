"""
Sample script to parse xr and ct radiology notes.
"""
import re
import numpy as np
import pandas as pd
import glob, os
import timeit

# Start timer
start_time = timeit.default_timer()

# Specify notes directory and columns to parse
notes_dir = input("What is the path to the notes?") # feel free to replace with the path to avoid being prompted every execution

notes = glob.glob(os.path.join(notes_dir, "*.txt"))
note_cols = ['ScanType', 'ScanDate', 'ClinInfo', 'Technique', 'Comparison', 'Findings', 'Impressions', 'ElecSig', 'RawText']

# Helper functions for parsing
def between(value, a, b):
    """
    Parse a substring between two other substrings.
    """
    # Find and validate before substring.
    pos_a = value.find(a)
    if pos_a == -1: return ""
    # Find and validate after substring.
    pos_b = value.rfind(b)
    if pos_b == -1: return ""
    # Return between substring.
    adjusted_pos_a = pos_a + len(a)
    if adjusted_pos_a >= pos_b: return ""
    return value[adjusted_pos_a:pos_b]

def before(value, a):
    """
    Parse a substring before (the first instance of) another substring.
    """
    # Find value string and return substring before it.
    pos_a = value.find(a)
    if pos_a == -1: return ""
    return value[0:pos_a]

def after(value, a):
    """
    Parse a substring after (the first instance of) another substring.
    """
    # Find and validate first part.
    pos_a = value.find(a)
    if pos_a == -1: return ""
    # Returns chars after the found string.
    adjusted_pos_a = pos_a + len(a)
    if adjusted_pos_a >= len(value): return ""
    return value[adjusted_pos_a:]


def parse_note(text):
    """
    Parse note and return array including each element.
    """
    scan_type = before(text, '**DATE<[').strip()
    scan_date = between(text, '**DATE<[**', 'CLINICAL INFORMATION: ').strip()
    clininfo = between(text, 'CLINICAL INFORMATION: ', ' TECHNIQUE: ').strip()
    technique = between(text, 'TECHNIQUE: ', ' COMPARISON: ').strip()
    comparison = between(text, 'COMPARISON: ', '\n\nFINDINGS:\n\n').strip()
    findings = between(text, '\n\nFINDINGS:\n\n', '\n\nIMPRESSION:\n\n').strip()
    impression = between(text, '\n\nIMPRESSION:\n\n', 'Report Electronically Signed:').strip()
    elec_sig = after(text, 'Report Electronically Signed: ').strip()
    raw_text = text
    return [scan_type, scan_date, clininfo, technique, comparison, findings, impression, elec_sig, raw_text]


def parsed_note_to_df(parsed_note_array):
    """
    Converts an array of note elements into a dataframe
    """
    temp_dict = dict(zip(note_cols, parsed_note_array))
    return pd.DataFrame(temp_dict, index = [0])



def main():
    """
    Parse notes and write to csv. 
    """
    notes_str = []

    for note in notes:
        with open(note, 'r') as file:
            notes_str.append(file.read())

    df_each_note = (parsed_note_to_df(parse_note(note)) for note in notes_str)
    df = pd.concat(df_each_note, ignore_index=True)
    df = df[note_cols]
    df.to_csv('notes_df.csv')


# Parse
main()
execution_time = timeit.default_timer() - start_time

print('Completed in ', str(format(execution_time, '.2f')), ' seconds.')

"""
Regex below works but might be slower. Can try with larger data set. Substitute this stuff into parse_note.

scan_type = text.partition('**DATE<[')[0].strip()
scan_date = re.search('DATE(.*) CLINICAL INFORMATION: ', text).group(1).strip()
clininfo = re.search('CLINICAL INFORMATION: (.*) TECHNIQUE: ', text).group(1).strip()
technique = re.search('TECHNIQUE: (.*) COMPARISON: ', text).group(1).strip()
comparison = re.search('COMPARISON: (.*)\n\nFINDINGS:\n\n', text).group(1).strip()
findings = re.search('\n\nFINDINGS:\n\n(.*)\n\nIMPRESSION:\n\n', text).group(1).strip()
impression = re.search('\n\nIMPRESSION:\n\n(.*)Report Electronically Signed:', text).group(1).strip()
elec_sig2 = text.partition('Report Electronically Signed: ')[2].strip()

# try catch if no match
"""