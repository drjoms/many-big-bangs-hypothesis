import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy import units as u
from astroquery.heasarc import Heasarc

OUTPUT_FILE = "Master_Evidence.csv"
MAX_MATCH_ARCSEC = 60
MAX_REDSHIFT_DELTA = 0.03


def find_column(table, candidates):
    normalized = {
        name.lower().replace("_", ""): name
        for name in table.colnames
    }
    for candidate in candidates:
        key = candidate.lower().replace("_", "")
        if key in normalized:
            return normalized[key]
    raise ValueError(
        f"Missing one of {candidates}; available columns: {table.colnames}"
    )


def numbers(table, column):
    return np.asarray(
        np.ma.filled(np.ma.asarray(table[column], dtype=float), np.nan),
        dtype=float,
    )


def text_values(table, column):
    values = np.ma.asarray(table[column])
    return np.asarray(np.ma.filled(values, ""), dtype=str)


def valid_text(values):
    return np.char.strip(values) != ""


def valid_row(name, ra, dec, mass, kt, redshift):
    return (
        bool(name.strip())
        and all(np.isfinite(x) for x in (ra, dec, mass, kt, redshift))
        and mass > 0
        and kt > 0
    )


def main():
    print("Querying HEASARC catalogs...")
    gcs = Heasarc.query_region(
        catalog="xmmsdssgcs", spatial="all-sky", columns="*"
    )
    xcs = Heasarc.query_region(
        catalog="xcs", spatial="all-sky", columns="*"
    )
    mcxc = Heasarc.query_region(
        catalog="mcxc", spatial="all-sky", columns="*"
    )

    rows = []

    # 1) XMMSDSSGCS: retain only sample 1 and its spectrally fitted kT_Ap.
    g_name = find_column(gcs, ["Name", "Alt_Name"])
    g_ra = find_column(gcs, ["RA"])
    g_dec = find_column(gcs, ["Dec", "DEC"])
    g_mass = find_column(gcs, ["Mass_500"])
    g_kt = find_column(gcs, ["kT_Ap"])
    g_z = find_column(gcs, ["Redshift"])
    g_sample = find_column(gcs, ["table_sample"])

    g_names = text_values(gcs, g_name)
    g_ras, g_decs = numbers(gcs, g_ra), numbers(gcs, g_dec)
    g_masses, g_kts = numbers(gcs, g_mass), numbers(gcs, g_kt)
    g_zs, g_samples = numbers(gcs, g_z), numbers(gcs, g_sample)

    for i in range(len(gcs)):
        if g_samples[i] != 1:
            continue  # sample 1 has fitted kT_Ap; exclude sample 2
        name = g_names[i].strip()
        mass_solar = g_masses[i]
        if not valid_row(name, g_ras[i], g_decs[i], mass_solar,
                         g_kts[i], g_zs[i]):
            continue

        rows.append({
            "Object_Name": name,
            "RA": g_ras[i],
            "DEC": g_decs[i],
            "Mass": mass_solar / 1e14,
            "Temp_Obs": g_kts[i],
            "Redshift": g_zs[i],
        })

    # 2) XCS: use its measured kT, but require a close MCXC mass match.
    m_ra = find_column(mcxc, ["RA"])
    m_dec = find_column(mcxc, ["Dec", "DEC"])
    m_mass = find_column(mcxc, ["Mass_500"])
    m_z = find_column(mcxc, ["Redshift"])

    m_ras, m_decs = numbers(mcxc, m_ra), numbers(mcxc, m_dec)
    m_masses, m_zs = numbers(mcxc, m_mass), numbers(mcxc, m_z)
    m_valid = (
        np.isfinite(m_ras) & np.isfinite(m_decs)
        & np.isfinite(m_masses) & (m_masses > 0)
        & np.isfinite(m_zs)
    )
    mcxc_coords = SkyCoord(
        ra=m_ras[m_valid] * u.deg,
        dec=m_decs[m_valid] * u.deg,
    )
    valid_mcxc_indices = np.flatnonzero(m_valid)

    x_name = find_column(xcs, ["Name"])
    x_ra = find_column(xcs, ["RA"])
    x_dec = find_column(xcs, ["Dec", "DEC"])
    x_kt = find_column(xcs, ["kT"])
    x_z = find_column(xcs, ["Redshift"])

    x_names = text_values(xcs, x_name)
    x_ras, x_decs = numbers(xcs, x_ra), numbers(xcs, x_dec)
    x_kts, x_zs = numbers(xcs, x_kt), numbers(xcs, x_z)

    x_valid = (
        valid_text(x_names)
        & np.isfinite(x_ras) & np.isfinite(x_decs)
        & np.isfinite(x_kts) & (x_kts > 0)
        & np.isfinite(x_zs)
    )
    x_indices = np.flatnonzero(x_valid)
    x_coords = SkyCoord(
        ra=x_ras[x_valid] * u.deg,
        dec=x_decs[x_valid] * u.deg,
    )

    match_indices, separations, _ = x_coords.match_to_catalog_sky(mcxc_coords)
    added_xcs = 0

    for j, x_i in enumerate(x_indices):
        m_i = valid_mcxc_indices[match_indices[j]]
        dz = abs(x_zs[x_i] - m_zs[m_i])

        if separations[j] > MAX_MATCH_ARCSEC * u.arcsec:
            continue
        if dz > MAX_REDSHIFT_DELTA * (1 + x_zs[x_i]):
            continue

        new_row = {
            "Object_Name": x_names[x_i].strip(),
            "RA": x_ras[x_i],
            "DEC": x_decs[x_i],
            "Mass": m_masses[m_i] / 1e14,
            "Temp_Obs": x_kts[x_i],  # copied from XCS; not calculated
            "Redshift": x_zs[x_i],
        }
        if not valid_row(
            new_row["Object_Name"], new_row["RA"], new_row["DEC"],
            new_row["Mass"], new_row["Temp_Obs"], new_row["Redshift"]
        ):
            continue

        # Avoid adding a second catalog entry for an existing same-position,
        # same-redshift cluster.
        duplicate = False
        for old in rows:
            sep = SkyCoord(
                ra=new_row["RA"] * u.deg, dec=new_row["DEC"] * u.deg
            ).separation(
                SkyCoord(ra=old["RA"] * u.deg, dec=old["DEC"] * u.deg)
            )
            if (
                sep <= MAX_MATCH_ARCSEC * u.arcsec
                and abs(new_row["Redshift"] - old["Redshift"])
                <= MAX_REDSHIFT_DELTA * (1 + new_row["Redshift"])
            ):
                duplicate = True
                break

        if not duplicate:
            rows.append(new_row)
            added_xcs += 1

    result = Table(rows=rows)
    result.write(OUTPUT_FILE, format="ascii.csv", overwrite=True)

    print(f"XMMSDSSGCS measured-temperature rows: {sum(1 for _ in rows) - added_xcs}")
    print(f"New XCS rows with MCXC mass matches: {added_xcs}")
    print(f"Total rows saved: {len(result)}")
    print(f"Saved to: {OUTPUT_FILE}")
    result[:5].pprint()


if __name__ == "__main__":
    main()

