
"""
Find optically red, quenched SDSS galaxies associated with
hot X-ray sources in the HEASARC XMMSDSSGCS catalogue.

Output:
    Dead_Hot_Galaxies_Evidence.csv

Install:
    pip install astroquery astropy numpy pandas
"""

import warnings
import numpy as np
import pandas as pd

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

from astroquery.heasarc import Heasarc
from astroquery.sdss import SDSS


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

X_RAY_CATALOG = "xmmsdssgcs"

MATCH_RADIUS = 20 * u.arcsec

# Illustrative selection thresholds; adjust for your science.
MIN_XRAY_TEMP_KEV = 2.0
MIN_U_R_COLOR = 2.2
MIN_G_R_COLOR = 0.7
MAX_HALPHA_EW_ANGSTROM = 3.0

OUTPUT_FILE = "Dead_Hot_Galaxies_Evidence.csv"

SDSS_RELEASE = 17

# Require a measured H-alpha equivalent width.
REQUIRE_HALPHA = True


# ---------------------------------------------------------
# GENERAL HELPERS
# ---------------------------------------------------------

def find_column(table, candidates):
    """Find a column without regard to capitalization."""

    lookup = {name.lower(): name for name in table.colnames}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


def clean_number(value):
    """Safely convert an astronomical table value to float."""

    try:
        if value is None or np.ma.is_masked(value):
            return np.nan

        result = float(value)

        if not np.isfinite(result):
            return np.nan

        return result

    except (ValueError, TypeError):
        return np.nan


def clean_text(value):
    """Safely convert a table value to text."""

    if value is None or np.ma.is_masked(value):
        return ""

    return str(value).strip()


# ---------------------------------------------------------
# 1. QUERY HEASARC
# ---------------------------------------------------------

def get_xray_catalog():

    print("Querying HEASARC:", X_RAY_CATALOG)

    # Only use actual aperture spectroscopic temperatures.
    query = f"""
        SELECT *
        FROM {X_RAY_CATALOG}
        WHERE kT_Ap IS NOT NULL
    """

    result = Heasarc.query_tap(query)

    if hasattr(result, "to_table"):
        result = result.to_table()

    if len(result) == 0:
        raise RuntimeError("No X-ray sources returned by HEASARC.")

    print(f"Retrieved {len(result)} X-ray records.")

    return result


# ---------------------------------------------------------
# 2. EXTRACT X-RAY COORDINATES AND TEMPERATURES
# ---------------------------------------------------------

def prepare_xray_data(table):

    ra_col = find_column(table, ["RA"])
    dec_col = find_column(table, ["Dec", "DEC"])
    temp_col = find_column(table, ["kT_Ap", "kT"])

    name_col = find_column(
        table, ["Name", "Short_Name", "DetID", "ObsID"]
    )

    lx_col = find_column(
        table, ["Lx_Ap", "Lx_500", "Cat_Lx"]
    )

    if not all([ra_col, dec_col, temp_col]):
        raise RuntimeError(
            "Required RA, DEC or spectroscopic temperature column missing."
        )

    records = []

    for row in table:

        ra = clean_number(row[ra_col])
        dec = clean_number(row[dec_col])
        temp = clean_number(row[temp_col])

        if not np.all(np.isfinite([ra, dec, temp])):
            continue

        if temp < MIN_XRAY_TEMP_KEV:
            continue

        name = (
            clean_text(row[name_col])
            if name_col else "Unnamed_Xray_Source"
        )

        lx = (
            clean_number(row[lx_col])
            if lx_col else np.nan
        )

        records.append({
            "Object_Name": name,
            "RA": ra,
            "DEC": dec,
            "Temp_Obs": temp,
            "L_X": lx
        })

    print(
        f"{len(records)} sources pass the "
        f"{MIN_XRAY_TEMP_KEV} keV temperature threshold."
    )

    return records


# ---------------------------------------------------------
# 3. QUERY SDSS AROUND EACH X-RAY SOURCE
# ---------------------------------------------------------

def get_sdss_match(ra, dec):

    xray_position = SkyCoord(
        ra=ra * u.deg,
        dec=dec * u.deg,
        frame="icrs"
    )

    try:

        matches = SDSS.query_region(
            xray_position,
            radius=MATCH_RADIUS,
            spectro=True,
            data_release=SDSS_RELEASE
        )

        if matches is None or len(matches) == 0:
            return None

        # Build SDSS coordinates.
        sdss_ra = np.array([
            clean_number(x) for x in matches["ra"]
        ])

        sdss_dec = np.array([
            clean_number(x) for x in matches["dec"]
        ])

        valid = np.isfinite(sdss_ra) & np.isfinite(sdss_dec)

        if not np.any(valid):
            return None

        matches = matches[valid]
        sdss_ra = sdss_ra[valid]
        sdss_dec = sdss_dec[valid]

        sdss_positions = SkyCoord(
            ra=sdss_ra * u.deg,
            dec=sdss_dec * u.deg,
            frame="icrs"
        )

        # Required spherical nearest-neighbour cross-match.
        index, separation, _ = (
            xray_position.match_to_catalog_sky(sdss_positions)
        )

        if separation > MATCH_RADIUS:
            return None

        return matches[index]

    except Exception as error:

        warnings.warn(
            f"SDSS query failed at RA={ra}, DEC={dec}: {error}"
        )

        return None


# ---------------------------------------------------------
# 4. RETRIEVE SDSS PHOTOMETRY AND H-ALPHA
# ---------------------------------------------------------

def get_sdss_properties(match):

    specid_col = find_column(match, ["specObjID"])

    if specid_col is None:
        return None

    specid = clean_text(match[specid_col])

    if not specid:
        return None

    try:

        # DR17 SpecPhotoAll supplies model magnitudes and redshift.
        # galSpecLine supplies emission-line measurements.
        sql = f"""
        SELECT TOP 1
            p.specObjID,
            p.z,
            p.class,
            p.modelMag_u,
            p.modelMag_g,
            p.modelMag_r,
            l.h_alpha_eqw

        FROM SpecPhotoAll AS p

        LEFT JOIN galSpecLine AS l
            ON p.specObjID = l.specObjID

        WHERE p.specObjID = {int(specid)}
        """

        result = SDSS.query_sql(
            sql,
            data_release=SDSS_RELEASE
        )

        if result is None or len(result) == 0:
            return None

        row = result[0]

        u_mag = clean_number(row["modelMag_u"])
        g_mag = clean_number(row["modelMag_g"])
        r_mag = clean_number(row["modelMag_r"])

        redshift = clean_number(row["z"])
        halpha = clean_number(row["h_alpha_eqw"])

        if not np.all(np.isfinite([u_mag, g_mag, r_mag])):
            return None

        u_r = u_mag - r_mag
        g_r = g_mag - r_mag

        return {
            "u_r": u_r,
            "g_r": g_r,
            "Halpha_EW": halpha,
            "Redshift": redshift,
            "SDSS_Class": clean_text(row["class"])
        }

    except Exception as error:

        warnings.warn(
            f"SDSS photometry/spectroscopy query failed: {error}"
        )

        return None


# ---------------------------------------------------------
# 5. IDENTIFY RED AND QUENCHED GALAXIES
# ---------------------------------------------------------

def is_quenched(properties):

    if properties is None:
        return False

    u_r = properties["u_r"]
    g_r = properties["g_r"]
    halpha = properties["Halpha_EW"]

    # Require a galaxy classification.
    if properties["SDSS_Class"].lower() != "galaxy":
        return False

    # Red optical colour.
    is_red = (
        u_r >= MIN_U_R_COLOR
        and g_r >= MIN_G_R_COLOR
    )

    if not is_red:
        return False

    # Fail closed if H-alpha is unavailable.
    if REQUIRE_HALPHA and not np.isfinite(halpha):
        return False

    if np.isfinite(halpha):
        if abs(halpha) > MAX_HALPHA_EW_ANGSTROM:
            return False

    return True


# ---------------------------------------------------------
# 6. MAIN PIPELINE
# ---------------------------------------------------------

def main():

    xray_table = get_xray_catalog()

    xray_sources = prepare_xray_data(xray_table)

    output = []

    for number, source in enumerate(xray_sources, start=1):

        print(
            f"[{number}/{len(xray_sources)}] "
            f"Processing {source['Object_Name']}"
        )

        match = get_sdss_match(
            source["RA"],
            source["DEC"]
        )

        if match is None:
            continue

        properties = get_sdss_properties(match)

        if not is_quenched(properties):
            continue

        output.append({

            "Object_Name": source["Object_Name"],

            "RA": source["RA"],
            "DEC": source["DEC"],

            "Temp_Obs": source["Temp_Obs"],

            "L_X": source["L_X"],

            "SDSS_Color_Index": properties["u_r"],

            "Redshift": properties["Redshift"]
        })

        print("  QUALIFIED: red + weak H-alpha + hot X-ray source")

    # -----------------------------------------------------
    # SAVE REAL OBSERVATIONAL RESULTS
    # -----------------------------------------------------

    columns = [
        "Object_Name",
        "RA",
        "DEC",
        "Temp_Obs",
        "L_X",
        "SDSS_Color_Index",
        "Redshift"
    ]

    dataframe = pd.DataFrame(output, columns=columns)

    dataframe.to_csv(
        OUTPUT_FILE,
        index=False,
        na_rep=""
    )

    print("\n------------------------------------")
    print("SEARCH COMPLETE")
    print("------------------------------------")

    print(f"Qualifying objects: {len(dataframe)}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":

    main()

