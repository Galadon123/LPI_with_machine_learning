import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Paths (relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parent
SPT_ROOT = PROJECT_ROOT / "SPT Value" / "SPT Value"
INPUT_TEMPLATE = PROJECT_ROOT / "Input value.xlsx"
OUTPUT_CSV = PROJECT_ROOT / "merged_spt.csv"
APPROVED_LOCATIONS_FILE = PROJECT_ROOT / "sylhet_division_locations.txt"

# --------------------------------------------------------
# 0.5. Soil Type Mapping (from tanveer-boltu-2.jpeg and tanveer-boltu-4.jpeg)
# --------------------------------------------------------

# Mapping: soil_type_key -> (USCS, FC_percent, FCI, gamma_kN_per_m3)
SOIL_TYPE_MAPPING = {
    # From tanveer-boltu-2.jpeg (FC table) and tanveer-boltu-4.jpeg (γ table)
    "reddish loose sand": ("SM", 20.0, 2, 16.0),
    "grey fine sand": ("SM", 18.0, 2, 17.5),
    "reddish sandy clay": ("CL", 70.0, 3, 19.5),
    "reddish fine sand": ("SM", 20.0, 2, 17.5),
    "grey silty clay": ("CL", 70.0, 3, 19.5),
    "soft clay": ("CL", 70.0, 3, 18.5),
    "grey sandy clay": ("CL", 70.0, 3, 19.5),
    "coarse sand": ("SP", 3.0, 1, 19.5),  # "if clean"
    "black organic": ("", np.nan, np.nan, 13.0),  # "if peat-like"
    # Additional variations that might appear:
    "reddish loose": ("SM", 20.0, 2, 16.0),
    "grey fine": ("SM", 18.0, 2, 17.5),
    "reddish fine": ("SM", 20.0, 2, 17.5),
    "reddish stiff clay": ("CL", 70.0, 3, 19.5),  # Stiff clay similar to sandy clay
    "grey hard clay": ("CL", 70.0, 3, 19.5),  # Hard clay similar to sandy clay
    "reddish soft clay": ("CL", 70.0, 3, 18.5),  # Soft clay variation
}


def match_soil_type(description: str):
    """
    Match a soil description string to a known soil type.
    Returns (soil_type_text, uscs, fc, fci, gamma) or (description, "", np.nan, np.nan, np.nan) if not found.
    """
    if not description or pd.isna(description):
        return ("", "", np.nan, np.nan, np.nan)
    
    desc_lower = str(description).strip().lower()
    # Remove trailing periods and extra spaces
    desc_lower = re.sub(r'\.+$', '', desc_lower).strip()
    
    # Try exact match first
    if desc_lower in SOIL_TYPE_MAPPING:
        uscs, fc, fci, gamma = SOIL_TYPE_MAPPING[desc_lower]
        return (description.strip(), uscs, fc, fci, gamma)
    
    # Try partial matching (e.g., "reddish soft clay" should match "soft clay")
    # Check if any key is contained in the description
    for key, (uscs, fc, fci, gamma) in SOIL_TYPE_MAPPING.items():
        if key in desc_lower or desc_lower in key:
            return (description.strip(), uscs, fc, fci, gamma)
    
    # Try matching key components (e.g., "reddish soft clay" -> match "soft clay")
    # Order by specificity (longer keys first)
    sorted_keys = sorted(SOIL_TYPE_MAPPING.keys(), key=len, reverse=True)
    for key in sorted_keys:
        # Check if all significant words in key are in description
        key_words = set(re.findall(r'\b\w+\b', key))
        desc_words = set(re.findall(r'\b\w+\b', desc_lower))
        # If most key words are present, consider it a match
        if len(key_words) > 0 and len(key_words.intersection(desc_words)) >= len(key_words) * 0.7:
            uscs, fc, fci, gamma = SOIL_TYPE_MAPPING[key]
            return (description.strip(), uscs, fc, fci, gamma)
    
    # No match found
    return (description.strip(), "", np.nan, np.nan, np.nan)

# --------------------------------------------------------
# 1. Load column structure from Input value.xlsx
# --------------------------------------------------------

template_df = pd.read_excel(INPUT_TEMPLATE)
TARGET_COLUMNS = [str(c) for c in template_df.columns]

# Remove legacy duplicate column if present (CRR is now Youd-style only).
_CRR_LEGACY_EXTRA = "CRR (Youd)"
if _CRR_LEGACY_EXTRA in TARGET_COLUMNS:
    TARGET_COLUMNS = [c for c in TARGET_COLUMNS if c != _CRR_LEGACY_EXTRA]
    template_df = template_df.drop(columns=[_CRR_LEGACY_EXTRA], errors="ignore")
    template_df.to_excel(INPUT_TEMPLATE, index=False)

# Factor of safety column FS = (CRR/CSR) * MSF; place after MSF.
_FS_COL = "FS"
if _FS_COL not in template_df.columns:
    _cols = list(template_df.columns)
    _ins = _cols.index("MSF") + 1 if "MSF" in _cols else len(_cols)
    template_df.insert(_ins, _FS_COL, np.nan)
    template_df.to_excel(INPUT_TEMPLATE, index=False)

# Liquefaction vs not (FS: >1 not liquefiable, <1 liquefiable; project sheet 4.25.10).
_LIQ_COL = "Liquefaction"
if _LIQ_COL not in template_df.columns:
    _cols = list(template_df.columns)
    _ins = _cols.index(_FS_COL) + 1 if _FS_COL in _cols else len(_cols)
    template_df.insert(_ins, _LIQ_COL, np.nan)
    template_df.to_excel(INPUT_TEMPLATE, index=False)

TARGET_COLUMNS = [str(c) for c in template_df.columns]

# --------------------------------------------------------
# 1.5. Load approved Sylhet Division locations whitelist
# --------------------------------------------------------

def load_approved_locations():
    """
    Load the list of approved Sylhet Division locations from file.
    Returns a set of location strings (normalized for matching).
    """
    approved = set()
    if APPROVED_LOCATIONS_FILE.exists():
        with open(APPROVED_LOCATIONS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip header lines, separators, and empty lines
                if not line or line.startswith('=') or line.startswith('Locations') or line.startswith('Total'):
                    continue
                # Extract location from numbered list format: "   1. Location Name"
                if '.' in line and line[0].isdigit():
                    parts = line.split('.', 1)
                    if len(parts) == 2:
                        loc = parts[1].strip()
                        if loc:
                            approved.add(loc)
                            # Also add normalized versions (without trailing dates/spaces)
                            normalized = loc.rstrip(' ,')
                            approved.add(normalized)
    return approved

APPROVED_LOCATIONS = load_approved_locations()
print(f"Loaded {len(APPROVED_LOCATIONS)} approved Sylhet Division location patterns")


# --------------------------------------------------------
# 2. Utility: parse info from file/folder names
# --------------------------------------------------------

def parse_location_and_borehole(file_path: Path):
    """
    Heuristic: folder name looks like
      'S-131 Ashis kumar das, Uttar Bagbari, 29.11.19(03)'
    and file names like 'Bore Chart-01.xls' or 'Bore Hole-02.xls'.
    Adjust this if your naming is different.
    """
    folder_name = file_path.parent.name
    file_name = file_path.name

    # Location string: everything after the first space (drop 'S-131')
    # e.g. 'Ashis kumar das, Uttar Bagbari, 29.11.19(03)'
    parts = folder_name.split(" ", 1)
    location = parts[1] if len(parts) > 1 else folder_name

    # Borehole number from file name (01, 02, 03, ...)
    bh_id = ""
    for token in file_name.replace(".", " ").replace("-", " ").split():
        if token.isdigit():
            bh_id = token
            break

    return location, bh_id


# --------------------------------------------------------
# 3. Geotechnical formulas (based on common practice)
#    Adjust them to match your JPEG formulas.
# --------------------------------------------------------

G = 9.81  # m/s^2

def compute_vertical_stresses(depth_m, unit_weight=18.0, gw_depth_m=None, gamma_w=9.81):
    """
    Simple layered assumption: same unit weight to depth.
    If gw_depth_m is given, assumes saturated below GWT with same gamma.
    You can refine with your own layering if needed.
    """
    if depth_m is None or np.isnan(depth_m):
        return np.nan, np.nan

    depth_m = float(depth_m)

    if gw_depth_m is None or np.isnan(gw_depth_m) or depth_m <= gw_depth_m:
        sigma_v = unit_weight * depth_m
        sigma_v_eff = sigma_v  # water table below
    else:
        # Above GWT
        sigma_above = unit_weight * gw_depth_m
        # Below GWT: effective unit weight approx (γ - γw)
        gamma_sat = unit_weight
        sigma_below_tot = gamma_sat * (depth_m - gw_depth_m)
        u = gamma_w * (depth_m - gw_depth_m)  # pore pressure
        sigma_v = sigma_above + sigma_below_tot
        sigma_v_eff = sigma_v - u

    return sigma_v, sigma_v_eff


def compute_cn(sigma_v_eff):
    """
    Overburden correction factor C_N from project formula:
        C_N = 0.77 * log10(2000 / σv')
    σv' = effective vertical stress (kPa), consistent with compute_vertical_stresses.
    """
    if sigma_v_eff is None or np.isnan(sigma_v_eff) or sigma_v_eff <= 0:
        return np.nan
    ratio = 2000.0 / float(sigma_v_eff)
    if ratio <= 0:
        return np.nan
    cn = 0.77 * math.log10(ratio)
    if np.isnan(cn) or cn <= 0:
        return np.nan
    return cn


def compute_rd(depth_m, mw=7.5):
    """
    Stress reduction factor rd (Youd et al. 2001 approximate).
    Depth in meters.
    """
    if depth_m is None or np.isnan(depth_m):
        return np.nan
    z = float(depth_m)
    # Basic expression for Mw=7.5
    return 1.0 - 0.015 * z


def compute_cr(depth_m):
    """
    Rod length correction C_R vs Depth-Z (m) below GL:
    (1–4) m → 0.75, (4–6) m → 0.85, (6–10) m → 0.95, >10 m → 1.0.
    For 0 < z < 1 m uses 0.75 (same as shallow bucket); z ≤ 0 → NaN.
    Boundaries: [4,6), [6,10), z ≥ 10 → 1.0; z < 4 → 0.75.
    """
    if depth_m is None or np.isnan(depth_m):
        return np.nan
    z = float(depth_m)
    if z <= 0:
        return np.nan
    if z < 4.0:
        return 0.75
    if z < 6.0:
        return 0.85
    if z < 10.0:
        return 0.95
    return 1.0


def compute_n60(n_raw, ce=0.6, cb=1.0, cs=1.0, cr=1.0):
    """
    N_60 = N * CE * CB * CS * CR
    """
    if n_raw is None or np.isnan(n_raw):
        return np.nan
    return n_raw * ce * cb * cs * cr


def compute_n1_60(n60, cn):
    """
    (N1)60 = N60 * CN
    """
    if (n60 is None or np.isnan(n60) or
            cn is None or np.isnan(cn)):
        return np.nan
    return n60 * cn


def round_n1_60_output(x):
    """
    Round (N1)60 for CSV output: nearest integer, half-up
    (e.g. 2.49 → 2, 2.5 → 3, 2.51 → 3). NaN unchanged.
    """
    if x is None or np.isnan(x):
        return np.nan
    return float(math.floor(float(x) + 0.5))


def compute_csr(a_max_g, sigma_v, sigma_v_eff, rd, mw=7.5):
    """
    Cyclic Stress Ratio for Mw=7.5 (Youd & Idriss, simplified):
    CSR = 0.65 * (a_max / g) * (σv / σv') * rd
    """
    if any(np.isnan(x) for x in [a_max_g, sigma_v, sigma_v_eff, rd]):
        return np.nan
    if sigma_v_eff <= 0:
        return np.nan
    return 0.65 * a_max_g * (sigma_v / sigma_v_eff) * rd


def compute_msf(mw):
    """
    Magnitude scaling factor (project sheet):
    MSF = 10^2.24 / M_w^2.56
    """
    if mw is None or np.isnan(mw) or float(mw) <= 0:
        return np.nan
    return (10.0 ** 2.24) / (float(mw) ** 2.56)


def row_is_valid_spt_depth_row(depth_ft):
    """
    Skip Excel artefact rows: zero-depth lines (often under the header) and
    blank trailing rows (non-finite depth). Real SPT intervals are > 0.
    """
    if depth_ft is None:
        return False
    try:
        d = float(depth_ft)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(d) or d <= 0:
        return False
    return True


def compute_fs(crr, csr, msf):
    """
    Factor of safety (project sheet): FS = (CRR / CSR) * MSF
    """
    if crr is None or csr is None or msf is None:
        return np.nan
    if any(isinstance(x, (float, np.floating)) and np.isnan(float(x)) for x in (crr, csr, msf)):
        return np.nan
    csr_f = float(csr)
    if csr_f == 0:
        return np.nan
    return (float(crr) / csr_f) * float(msf)


def liquefaction_class_from_fs(fs):
    """
    From project sheet: FS > 1 → not liquefiable; FS < 1 → liquefiable.
    FS == 1 → at limit; missing FS → NaN.
    """
    if fs is None or (isinstance(fs, (float, np.floating)) and np.isnan(float(fs))):
        return np.nan
    f = float(fs)
    if f > 1.0:
        return "Not liquefiable"
    if f < 1.0:
        return "Liquefiable"
    return "At limit (FS=1)"


def compute_alpha_beta_fc(fc):
    """
    Fines-content factors for (N1)60cs = α + β (N1)60 (Youd et al. simplified / project sheets).
    FC in percent. Missing FC → treat as clean sand (FC ≤ 5%).
    """
    if fc is None or (isinstance(fc, float) and np.isnan(fc)):
        fc = 0.0
    fc = float(fc)
    if fc <= 5.0:
        return 0.0, 1.0
    if fc < 35.0:
        alpha = math.exp(1.76 - 190.0 / (fc * fc))
        beta = 0.99 + (fc ** 1.5) / 1000.0
        return alpha, beta
    return 5.0, 1.2


def compute_n1_60_cs_youd(n1_60, fc):
    """(N1)60,cs = α + β · (N1)60"""
    if n1_60 is None or np.isnan(n1_60):
        return np.nan
    alpha, beta = compute_alpha_beta_fc(fc)
    return alpha + beta * float(n1_60)


def compute_crr_youd_from_n1_60_cs(n1_60_cs):
    """
    CRR from project sheet (exp form), N = (N1)60,cs:
    CRR = exp( N/14.1 + (N/126)² - (N/23.6)³ + (N/25.4)⁴ - 2.8 )
    """
    if n1_60_cs is None or np.isnan(n1_60_cs):
        return np.nan
    n = float(n1_60_cs)
    if n < 0:
        return np.nan
    try:
        p = (
            n / 14.1
            + (n / 126.0) ** 2
            - (n / 23.6) ** 3
            + (n / 25.4) ** 4
            - 2.8
        )
        crr = math.exp(p)
    except OverflowError:
        return np.nan
    if math.isinf(crr) or math.isnan(crr):
        return np.nan
    return float(crr)


def compute_crr_youd_column(n1_60_continuous, fc):
    """
    CRR column: (N1)60,cs = α + β·(N1)60 (FC per project sheet), then exp CRR above.
    Uses continuous (N1)60 before rounding.
    """
    if n1_60_continuous is None or np.isnan(n1_60_continuous):
        return np.nan
    n1 = float(n1_60_continuous)
    n1_cs = compute_n1_60_cs_youd(n1, fc)
    return compute_crr_youd_from_n1_60_cs(n1_cs)


# --------------------------------------------------------
# 4. Extract rows from a single bore chart Excel file
# --------------------------------------------------------

def extract_gwt_from_sheet(df: pd.DataFrame):
    """
    Extract Ground Water Level (GWT) from Excel sheet.
    Looks for pattern like "GROUND WATER LEVEL : 11'-0" BELOW EGL."
    Returns GWT in meters, or None if not found.
    """
    df_str = df.astype(str)
    
    # Search for "GROUND WATER LEVEL" text
    mask = df_str.apply(lambda col: col.str.contains("GROUND WATER LEVEL", case=False, na=False))
    if not mask.any().any():
        return None
    
    # Find the cell containing GWT info
    r_idx, c_idx = list(zip(*mask.values.nonzero()))[0]
    
    # Check the cell with GWT label and adjacent cells (in case value is split)
    gwt_texts = []
    for dr in [0]:  # Same row
        for dc in [0, 1, 2]:  # Current cell and next 2 columns
            try:
                cell_val = str(df.iloc[r_idx + dr, c_idx + dc])
                if cell_val and cell_val != 'nan':
                    gwt_texts.append(cell_val)
            except (IndexError, KeyError):
                pass
    
    # Combine all text and search for GWT value
    combined_text = ' '.join(gwt_texts)
    
    # Parse patterns like:
    # "GROUND WATER LEVEL : 11'-0" BELOW EGL."
    # "GROUND WATER LEVEL : 9'-0" BELOW EGL."
    # "GROUND WATER LEVEL : 1.5m BELOW EGL."
    # "11'-0"" (just the value)
    
    # Try to extract feet value (pattern: number followed by '- or ' or "-0")
    # More specific: look for pattern like "11'-0" or "11'" or "11'-"
    match_ft = re.search(r'(\d+(?:\.\d+)?)\s*[\'-]\s*(?:0["\'])?', combined_text)
    if match_ft:
        gwt_ft = float(match_ft.group(1))
        # Sanity check: GWT should be reasonable (between 0 and 50 feet typically)
        if 0 <= gwt_ft <= 50:
            return gwt_ft * 0.3048  # Convert feet to meters
    
    # Try to extract meters value (pattern: number followed by 'm')
    match_m = re.search(r'(\d+(?:\.\d+)?)\s*m\b', combined_text, re.IGNORECASE)
    if match_m:
        gwt_m = float(match_m.group(1))
        # Sanity check: GWT should be reasonable (between 0 and 15 meters typically)
        if 0 <= gwt_m <= 15:
            return gwt_m
    
    return None


def extract_rows_from_bore_chart(df: pd.DataFrame, file_path: Path):
    """
    Map one bore chart sheet -> list of rows for merged_spt.

    You MUST adjust:
      - which columns hold Depth, N-06, N-12, N-18, soil descriptions, etc.
      - any special rules you have (from JPEG formulas).
    """
    rows = []

    # Extract GWT from the sheet first
    gwt_m = extract_gwt_from_sheet(df)
    if gwt_m is None:
        gwt_m = 1.5  # Default fallback

    # ------------------------------------------------------------------
    # SPECIAL CASE: Bore Chart tables with explicit SPT block:
    #
    #   ... | SPT Intervals(ft) | 06 Inch | 06 Inch | 06 Inch | N- Value | Layer Change | ...
    #
    # We treat these as:
    #   Depth(ft) = SPT Intervals(ft)
    #   N-06      = first  06 Inch
    #   N-12      = second 06 Inch
    #   N-18      = third  06 Inch
    #   N -Value  = N-  Value  (already sum of last two 06 columns)
    # ------------------------------------------------------------------
    df_no_header = df.copy()
    # Ensure columns are a simple RangeIndex so integer positions
    # (c_idx) can be used directly.
    df_no_header.columns = range(df_no_header.shape[1])

    mask = df_no_header.astype(str).apply(
        lambda col: col.str.contains("SPT Intervals", case=False, na=False)
    )
    if mask.any().any():
        (r_idx, c_idx) = list(zip(*mask.values.nonzero()))[0]

        # Expect specific header pattern starting at this row
        header_row = df_no_header.iloc[r_idx]
        # Basic sanity check to avoid mis-detecting random text
        if (
            str(header_row[c_idx]).strip().lower().startswith("spt intervals")
            and str(header_row.get(c_idx + 1, "")).strip().lower().startswith("06")
            and str(header_row.get(c_idx + 2, "")).strip().lower().startswith("06")
            and str(header_row.get(c_idx + 3, "")).strip().lower().startswith("06")
        ):
            depth_col_idx = c_idx
            six1_idx = c_idx + 1
            six2_idx = c_idx + 2
            six3_idx = c_idx + 3
            nvalue_idx = c_idx + 4

            # Find "DESCRIPTION OF SOIL STRATA" column
            soil_desc_idx = None
            for col_idx in range(df_no_header.shape[1]):
                cell_val = str(df_no_header.iloc[r_idx, col_idx]).strip().upper()
                if "DESCRIPTION" in cell_val and "SOIL" in cell_val:
                    soil_desc_idx = col_idx
                    break

            # Info from file/folder name
            location_str, borehole_id = parse_location_and_borehole(file_path)

            a_max = 3.5316
            a_max_g = a_max / G
            mw = 7.5
            gamma_w = 9.81
            gwt_m_default = gwt_m  # Use extracted GWT instead of hardcoded 1.5

            # Track last known soil type to propagate forward
            last_soil_type = ""
            last_soil_uscs = ""
            last_fc = float("nan")
            last_fci = float("nan")
            last_gamma = 18.0  # Default unit weight

            for i in range(r_idx + 1, len(df_no_header)):
                row_raw = df_no_header.iloc[i]
                depth_ft = row_raw.get(depth_col_idx)
                try:
                    depth_ft = float(depth_ft)
                except (TypeError, ValueError):
                    # Stop when we reach bottom / blank rows
                    continue

                if not row_is_valid_spt_depth_row(depth_ft):
                    continue

                depth_m = depth_ft * 0.3048

                def _num(idx):
                    try:
                        v = row_raw.get(idx)
                        return float(v) if v is not None and v == v else float("nan")
                    except Exception:
                        return float("nan")

                n06 = _num(six1_idx)
                n12 = _num(six2_idx)
                n18 = _num(six3_idx)
                n_total = _num(nvalue_idx)

                # Extract soil description and map to properties
                soil_desc = ""
                if soil_desc_idx is not None:
                    soil_desc_raw = row_raw.get(soil_desc_idx)
                    if soil_desc_raw is not None and str(soil_desc_raw).strip() and str(soil_desc_raw) != 'nan':
                        soil_desc = str(soil_desc_raw).strip()
                
                # Try to match soil type from current row
                soil_type_text, soil_uscs, fc, fci, gamma = match_soil_type(soil_desc)
                
                # If current row has a valid soil type, use it and update last known values
                if soil_type_text and soil_uscs:
                    # Valid soil type found - use it and remember for next rows
                    last_soil_type = soil_type_text
                    last_soil_uscs = soil_uscs
                    last_fc = fc if not pd.isna(fc) else float("nan")
                    last_fci = fci if not pd.isna(fci) else float("nan")
                    last_gamma = gamma if not pd.isna(gamma) and gamma > 0 else 18.0
                elif soil_desc:
                    # Has description but no match - use description as-is, keep last properties
                    soil_type_text = soil_desc
                    soil_uscs = last_soil_uscs
                    fc = last_fc
                    fci = last_fci
                    gamma = last_gamma
                else:
                    # No soil description - propagate last known values
                    soil_type_text = last_soil_type
                    soil_uscs = last_soil_uscs
                    fc = last_fc
                    fci = last_fci
                    gamma = last_gamma

                sigma_v, sigma_v_eff = compute_vertical_stresses(
                    depth_m, unit_weight=gamma, gw_depth_m=gwt_m_default, gamma_w=gamma_w
                )
                cn = compute_cn(sigma_v_eff)
                rd = compute_rd(depth_m, mw=mw)

                ce = 0.6
                cb = 1.0
                cs = 1.0
                cr = compute_cr(depth_m)

                n60 = compute_n60(n_total, ce=ce, cb=cb, cs=cs, cr=cr)
                n1_60_continuous = compute_n1_60(n60, cn)
                n1_60 = round_n1_60_output(n1_60_continuous)

                csr_7_5 = compute_csr(a_max_g, sigma_v, sigma_v_eff, rd, mw=mw)
                msf = compute_msf(mw)
                crr_7_5 = compute_crr_youd_column(n1_60_continuous, fc)
                fs_val = compute_fs(crr_7_5, csr_7_5, msf)
                liq_class = liquefaction_class_from_fs(fs_val)

                sl = float("nan")
                rp = float("nan")

                rows.append(
                    {
                        "Location": location_str,
                        "Lat,Long": "",
                        "Borehole ": borehole_id,
                        "Depth(ft)": depth_ft,
                        "Depth-Z(m)": depth_m,
                        "GWT(m)": gwt_m_default,
                        "N-06": n06,
                        "N-12": n12,
                        "N-18": n18,
                        "N -Value": n_total,
                        "σv": sigma_v,
                        "σv'": sigma_v_eff,
                        "CE": ce,
                        "CB": cb,
                        "CS": cs,
                        "CR": cr,
                        "CN": cn,
                        "(N1)60": n1_60,
                        "Soil Type ": soil_type_text,
                        "Soil USCS": soil_uscs,
                        "FC": fc,
                        "FCI": fci,
                        "rd": rd,
                        "CSR7.5": csr_7_5,
                        "CRR": crr_7_5,
                        "Rp": rp,
                        "a(max)": a_max,
                        "g": G,
                        "Mw": mw,
                        "MSF": msf,
                        "FS": fs_val,
                        "Liquefaction": liq_class,
                        "SL": sl,
                        "γ": gamma,
                        "γw": gamma_w,
                    }
                )

            return rows

    # ------------------------------------------------------------------
    # GENERIC FALLBACK (older/other formats)
    # ------------------------------------------------------------------
    # ---- 4.1. Identify columns in this sheet (EDIT THESE) ----
    # If your sheet has headers, read with header=0 in read_excel and match by name.
    # If not, match by column index (0,1,2,...). Example below assumes headers:

    df = df.rename(columns=lambda c: str(c).strip())

    # Lowercase map for flexible header matching
    lower_cols = {c: c.lower() for c in df.columns}

    # Try to find depth column by common header names first (very tolerant).
    depth_candidates = []
    for c, lc in lower_cols.items():
        if (
            "depth" in lc           # e.g. "Depth(ft)", "DEPTH", "Depth-Z(m)"
            or "z(m" in lc          # e.g. "Z(m)"
            or "depth-z" in lc
        ):
            depth_candidates.append(c)

    depth_col = depth_candidates[0] if depth_candidates else None

    # --- N-value columns ---
    # 1) Prefer explicit N-06/N-12/N-18, if present.
    n06_col = next((c for c in df.columns if "n-06" in lower_cols[c] or "n 06" in lower_cols[c]), None)
    n12_col = next((c for c in df.columns if "n-12" in lower_cols[c] or "n 12" in lower_cols[c]), None)
    n18_col = next((c for c in df.columns if "n-18" in lower_cols[c] or "n 18" in lower_cols[c]), None)

    # 2) If not present, many of your Bore Chart-01.xls files have three
    #    columns each titled something like '06', '06', '06' and a final
    #    N-value column which is the sum of the LAST TWO '06' columns.
    #    We detect that pattern and map them to N-06, N-12, N-18.
    n_value_col = next((c for c in df.columns if "n-value" in lower_cols[c] or "n value" in lower_cols[c]), None)

    if n06_col is None and n12_col is None and n18_col is None:
        six_cols = []
        for c, lc in lower_cols.items():
            header = c.strip().lower()
            if header == "06" or header.startswith("06 ") or " 06 " in header or "0-6" in header:
                six_cols.append(c)
        if len(six_cols) >= 3:
            # Take first three 06-columns in left-to-right order
            six_cols = six_cols[:3]
            n06_col, n12_col, n18_col = six_cols[0], six_cols[1], six_cols[2]

    # Find soil description column - prefer "DESCRIPTION OF SOIL STRATA" or similar
    soil_col = None
    for c in df.columns:
        lc = str(c).lower()
        if "description" in lc and "soil" in lc:
            soil_col = c
            break
    # Fallback to any column with "Soil" or "Description"
    if soil_col is None:
        soil_col = next((c for c in df.columns if "Soil" in str(c) or "Description" in str(c)), None)

    if depth_col is None:
        # Fallback: for Bore Chart / Bore Hole files with no explicit header,
        # assume the **first numeric column** is depth.
        fname_lower = file_path.name.lower()
        if "bore" in fname_lower:
            first_numeric_col = None
            for c in df.columns:
                series = df[c]
                # Check if this column is mostly numeric
                numeric_fraction = pd.to_numeric(series, errors="coerce").notna().mean()
                if numeric_fraction > 0.5:
                    first_numeric_col = c
                    break

            if first_numeric_col is not None:
                depth_col = first_numeric_col
            else:
                print(f"Skipping {file_path} (no depth column found even after numeric scan)")
                return rows
        else:
            # Non-bore files (e.g. layout sheets) are safe to skip
            print(f"Skipping {file_path} (no depth column found)")
            return rows

    # ---- 4.2. Info from file/folder name ----
    location_str, borehole_id = parse_location_and_borehole(file_path)

    # Project-wide constants (adjust to your JPEG values if different)
    a_max = 3.5316  # peak ground acceleration (m/s^2)
    a_max_g = a_max / G
    mw = 7.5
    gamma_w = 9.81

    # GWT was already extracted from the sheet at the beginning of this function
    # Use extracted value, fallback to 1.5m if not found
    gwt_m_default = gwt_m if gwt_m is not None else 1.5

    # Track last known soil type to propagate forward
    last_soil_type = ""
    last_soil_uscs = ""
    last_fc = np.nan
    last_fci = np.nan
    last_gamma = 18.0  # Default unit weight

    for _, r in df.iterrows():
        depth_ft = r.get(depth_col)
        # Skip blank cells
        if depth_ft is None or pd.isna(depth_ft):
            continue

        # Some sheets keep the header text (e.g. "SPT Intervals(ft)") in the
        # same column; skip any non‑numeric depth values.
        try:
            depth_ft = float(depth_ft)
        except (TypeError, ValueError):
            continue

        if not row_is_valid_spt_depth_row(depth_ft):
            continue

        # ---- Depth & basic N-value handling ----
        depth_m = depth_ft * 0.3048

        n06 = float(r.get(n06_col)) if n06_col and not pd.isna(r.get(n06_col)) else np.nan
        n12 = float(r.get(n12_col)) if n12_col and not pd.isna(r.get(n12_col)) else np.nan
        n18 = float(r.get(n18_col)) if n18_col and not pd.isna(r.get(n18_col)) else np.nan

        # Total N:
        # - If we are in the 3×'06' pattern, project rule says:
        #       N-value = (second 06) + (third 06)  => n12 + n18
        # - Otherwise, fall back to sum of available components.
        if not np.isnan(n12) or not np.isnan(n18):
            n_total = (0 if np.isnan(n12) else n12) + (0 if np.isnan(n18) else n18)
        elif not np.isnan(n06):
            n_total = n06
        else:
            n_total = np.nan

        # Extract soil description and map to properties
        soil_desc_raw = str(r.get(soil_col)) if soil_col and not pd.isna(r.get(soil_col)) else ""
        soil_type_text, soil_uscs, fc, fci, gamma = match_soil_type(soil_desc_raw)
        
        # If current row has a valid soil type, use it and update last known values
        if soil_type_text and soil_uscs:
            # Valid soil type found - use it and remember for next rows
            last_soil_type = soil_type_text
            last_soil_uscs = soil_uscs
            last_fc = fc if not pd.isna(fc) else np.nan
            last_fci = fci if not pd.isna(fci) else np.nan
            last_gamma = gamma if not pd.isna(gamma) and gamma > 0 else 18.0
        elif soil_desc_raw:
            # Has description but no match - use description as-is, keep last properties
            soil_type_text = soil_desc_raw
            soil_uscs = last_soil_uscs
            fc = last_fc
            fci = last_fci
            gamma = last_gamma
        else:
            # No soil description - propagate last known values
            soil_type_text = last_soil_type
            soil_uscs = last_soil_uscs
            fc = last_fc
            fci = last_fci
            gamma = last_gamma

        # ---- Stresses & corrections ----
        sigma_v, sigma_v_eff = compute_vertical_stresses(depth_m, unit_weight=gamma,
                                                         gw_depth_m=gwt_m_default,
                                                         gamma_w=gamma_w)
        cn = compute_cn(sigma_v_eff)
        rd = compute_rd(depth_m, mw=mw)

        # Hammer correction etc. – set to values from your Input value/JPEG
        ce = 0.6
        cb = 1.0
        cs = 1.0
        cr = compute_cr(depth_m)

        n60 = compute_n60(n_total, ce=ce, cb=cb, cs=cs, cr=cr)
        n1_60_continuous = compute_n1_60(n60, cn)
        n1_60 = round_n1_60_output(n1_60_continuous)

        csr_7_5 = compute_csr(a_max_g, sigma_v, sigma_v_eff, rd, mw=mw)
        msf = compute_msf(mw)
        crr_7_5 = compute_crr_youd_column(n1_60_continuous, fc)
        fs_val = compute_fs(crr_7_5, csr_7_5, msf)
        liq_class = liquefaction_class_from_fs(fs_val)

        # You may have your own definition of SL, Rp, etc.
        sl = np.nan
        rp = np.nan

        row = {
            "Location": location_str,
            "Lat,Long": "",
            "Borehole ": borehole_id,
            "Depth(ft)": depth_ft,
            "Depth-Z(m)": depth_m,
            "GWT(m)": gwt_m_default,
            "N-06": n06,
            "N-12": n12,
            "N-18": n18,
            "N -Value": n_total,
            "σv": sigma_v,
            "σv'": sigma_v_eff,
            "CE": ce,
            "CB": cb,
            "CS": cs,
            "CR": cr,
            "CN": cn,
            "(N1)60": n1_60,
            "Soil Type ": soil_type_text,
            "Soil USCS": soil_uscs,
            "FC": fc,
            "FCI": fci,
            "rd": rd,
            "CSR7.5": csr_7_5,
            "CRR": crr_7_5,
            "Rp": rp,
            "a(max)": a_max,
            "g": G,
            "Mw": mw,
            "MSF": msf,
            "FS": fs_val,
            "Liquefaction": liq_class,
            "SL": sl,
            "γ": gamma,
            "γw": gamma_w,
        }

        # Keep only columns that exist in TARGET_COLUMNS; fill missing ones with NaN later
        rows.append(row)

    return rows


# --------------------------------------------------------
# 5. Main: walk all SPT Value files, merge, and write merged_spt.csv
# --------------------------------------------------------

def is_location_approved(location_str):
    """
    Check if a location string matches any approved Sylhet Division location.
    Handles variations in formatting (dates, parentheses, etc.)
    """
    if not APPROVED_LOCATIONS:
        return True  # If no whitelist loaded, allow all
    
    location_clean = location_str.strip().rstrip(' ,')
    
    # Direct match
    if location_clean in APPROVED_LOCATIONS:
        return True
    
    # Remove trailing date patterns like "(03)" or ", 18.01.2020(03)"
    location_no_date = re.sub(r'\s*\([^)]+\)\s*$', '', location_clean)
    location_no_date = location_no_date.rstrip(' ,')
    
    if location_no_date in APPROVED_LOCATIONS:
        return True
    
    # Check if any approved location is contained in this location (or vice versa)
    # This handles cases where folder name has extra info
    for approved_loc in APPROVED_LOCATIONS:
        approved_clean = approved_loc.strip().rstrip(' ,')
        # Remove dates from approved location too
        approved_no_date = re.sub(r'\s*\([^)]+\)\s*$', '', approved_clean)
        approved_no_date = approved_no_date.rstrip(' ,')
        
        # Check if one contains the other (for partial matches)
        if len(approved_no_date) > 15 and len(location_no_date) > 15:
            if approved_no_date in location_no_date or location_no_date in approved_no_date:
                return True
    
    return False


def main():
    all_rows = []
    skipped_count = 0

    for xls_path in SPT_ROOT.rglob("*.xls"):
        # Extract location from file path to check if it's approved
        location_str, _ = parse_location_and_borehole(xls_path)
        
        # Check if this location is in the approved Sylhet Division list
        if not is_location_approved(location_str):
            skipped_count += 1
            continue
        
        # Skip obvious junk files if needed (e.g. by name or via your merged_spt_errors list)
        try:
            df = pd.read_excel(xls_path, sheet_name=0)
        except Exception as e:
            print(f"ERROR reading {xls_path}: {e}")
            continue

        rows = extract_rows_from_bore_chart(df, xls_path)
        if rows:
            all_rows.extend(rows)
    
    print(f"Skipped {skipped_count} files from locations outside Sylhet Division")

    if not all_rows:
        print("No rows extracted. Check column mappings in extract_rows_from_bore_chart().")
        return

    merged = pd.DataFrame(all_rows)

    # Ensure all TARGET_COLUMNS exist; if missing, add as NaN
    for col in TARGET_COLUMNS:
        if col not in merged.columns:
            merged[col] = np.nan

    # Reorder columns to match Input value.xlsx
    merged = merged[TARGET_COLUMNS]

    n_before = len(merged)
    merged = merged.drop_duplicates()
    _key = [c for c in ("Location", "Borehole ", "Depth(ft)") if c in merged.columns]
    if _key:
        merged = merged.drop_duplicates(subset=_key, keep="first")
    n_after = len(merged)
    if n_after < n_before:
        print(f"Removed {n_before - n_after} duplicate rows ({n_before} -> {n_after})")

    merged.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"Wrote {len(merged)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()