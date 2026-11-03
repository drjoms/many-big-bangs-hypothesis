import numpy as np
from astropy.table import Table
from astroquery.heasarc import Heasarc

OUTPUT_FILE = "Master_Evidence.csv"


def find_column(table, candidates):
    """Find a column name case-insensitively, ignoring underscores."""
    by_normalized_name = {
        name.lower().replace("_", ""): name
        for name in table.colnames
    }
    for candidate in candidates:
        key = candidate.lower().replace("_", "")
        if key in by_normalized_name:
            return by_normalized_name[key]

    raise ValueError(
        f"Could not find any of {candidates}. "
        f"Available columns: {table.colnames}"
    )


def valid_numeric(column):
    values = np.ma.asarray(column, dtype=float)
    return (
        ~np.ma.getmaskarray(values)
        & np.isfinite(np.ma.filled(values, np.nan))
    )


def valid_text(column):
    values = np.ma.asarray(column)
    mask = np.ma.getmaskarray(values)
    text = np.asarray(np.ma.filled(values, ""), dtype=str)
    return ~mask & (np.char.strip(text) != "")


def main():
    print("Querying HEASARC xmmsdssgcs...")
    data = Heasarc.query_region(
        catalog="xmmsdssgcs",
        spatial="all-sky",
        columns="*",
    )

    name_col = find_column(data, ["Name", "Alt_Name"])
    ra_col = find_column(data, ["RA"])
    dec_col = find_column(data, ["Dec", "DEC"])
    mass_col = find_column(data, ["Mass_500"])
    kt_col = find_column(data, ["kT_Ap"])  # Measured spectral-fit temperature
    redshift_col = find_column(data, ["Redshift"])

    # Require a name and valid values for every field in the requested CSV.
    keep = (
        valid_text(data[name_col])
        & valid_numeric(data[ra_col])
        & valid_numeric(data[dec_col])
        & valid_numeric(data[mass_col])
        & valid_numeric(data[kt_col])
        & valid_numeric(data[redshift_col])
    )

    result = Table()
    result["Object_Name"] = np.asarray(data[name_col][keep], dtype=str)
    result["RA"] = data[ra_col][keep]
    result["DEC"] = data[dec_col][keep]
    # The catalog's Mass_500 is in solar masses; convert to 10^14 solar masses.
    result["Mass"] = data[mass_col][keep] / 1e14
    # Copy the catalog's measured kT_Ap values; no temperature is calculated.
    result["Temp_Obs"] = data[kt_col][keep]
    result["Redshift"] = data[redshift_col][keep]

    result.write(OUTPUT_FILE, format="ascii.csv", overwrite=True)

    print(f"Rows fetched: {len(data)}")
    print(f"Rows retained with valid measured kT_Ap and mass: {len(result)}")
    print(f"Saved to: {OUTPUT_FILE}")
    print("First five rows:")
    result[:5].pprint()


if __name__ == "__main__":
    main()

