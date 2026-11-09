(.venv) dimko@dimkosPC ~/work/physics_experiments/gather_residial_heat/hydro_temp_mass_gemma/try1_gpt $ cat masstotempMK18.py 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
import sys
import argparse
from scipy.stats import linregress

# Versioning
__version__ = "10.0.0-SLOPE_DISCOVERY_ENGINE"

# =================================================================
# CORE PHYSICS
# =================================================================

def hardness_to_temp(hr):
    """Maps X-ray Hardness Ratio (EP_HR1) to Temperature (keV)."""
    return 0.1 * (1 + hr * 10)

# =================================================================
# MAIN ANALYSIS ENGINE
# =================================================================

def main(file_a, file_b, mass_col='SERSIC_MASS', temp_col='EP_HR1', radius_arcsec=1.0):
    print(f"\n{'='*75}")
    print(f" ISP LOCAL SLOPE DISCOVERY | v{__version__}")
    print(f"{'='*75}\n")

    # 1. DATA INGESTION & SPATIAL MATCHING
    print(f"[1/4] INGESTING & CROSS-MATCHING...")
    try:
        with fits.open(file_a) as h1, fits.open(file_b) as h2:
            df_a = Table(h1[1].data).to_pandas()
            df_b = Table(h2[1].data).to_pandas()
    except Exception as e:
        print(f"CRITICAL ERROR: {e}"); return

    def find_coords(df):
        ra = next((c for c in df.columns if 'RA' in c.upper()), None)
        dec = next((c for c in df.columns if 'DEC' in c.upper()), None)
        return ra, dec

    ra_a, dec_a = find_coords(df_a)
    ra_b, dec_b = find_coords(df_b)

    if not all([ra_a, dec_a, ra_b, dec_b]):
        print("CRITICAL ERROR: RA/DEC columns not found."); return

    coords_a = SkyCoord(ra=df_a[ra_a].values*u.degree, dec=df_a[dec_a].values*u.degree)
    coords_b = SkyCoord(ra=df_b[ra_b].values*u.degree, dec=df_b[dec_b].values*u.degree)
    result = coords_a.match_to_catalog_sky(coords_b)
    mask = result[1] <= (radius_arcsec * u.arcsec)
    
    m_matched = df_a.iloc[np.where(mask)[0]][mass_col].values.astype(float)
    t_raw = df_b.iloc[result[0][mask]][temp_col].values.astype(float)
    T_matched = hardness_to_temp(t_raw)

    # Filter for strictly positive (Origin handling)
    valid = (m_matched > 1e-5) & (T_matched > 1e-8)
    M = m_matched[valid]
    T = T_matched[valid]
    
    # Sort by mass
    sort_idx = np.argsort(M)
    M = M[sort_idx]
    T = T[sort_idx]

    print(f"      SUCCESS: {len(M)} synchronized sources matched.\n")

    # 2. LOCAL SLOPE CALCULATION (The Core Engine)
    # We use a sliding window to find the local 'alpha' at every mass point.
    print(f"[2/4] CALCULATING LOCAL EXPONENTS (alpha)...")
    
    log_M = np.log10(M)
    log_T = np.log10(T)
    
    local_alphas = []
    # Window size: number of neighbors to use for the local slope
    window = max(5, int(len(M)**0.3)) 

    for i in range(len(M)):
        # Get a neighborhood around the current point
        start = max(0, i - window // 2)
        end = min(len(M), i + window // 2)
        
        # If the window is too small at the edges, expand it
        if (end - start) < 3:
            start = 0
            end = min(len(M), window * 2)

        win_log_m = log_M[start:end]
        win_log_t = log_T[start:end]
        
        # Calculate the local slope (alpha)
        slope, _, _, _, _ = linregress(win_log_m, win_log_t)
        local_alphas.append(slope)
    
    local_alphas = np.array(local_alphas)

    # 3. STATISTICAL BREAKDOWN
    print(f"[3/4] ANALYZING EXPONENT TREND...")
    
    # Compare the bottom 25% to the top 25%
    split_idx = int(len(M) * 0.25)
    alpha_low = np.mean(local_alphas[:split_idx])
    alpha_high = np.mean(local_alphas[-split_idx:])
    
    # Count how many points are significantly above 2/3
    isp_count = np.sum(local_alphas > 0.6666)
    standard_count = np.sum(local_alphas <= 0.6666)

    print(f"\n{'='*50}")
    print(f" EXPONENT TREND SUMMARY")
    print(f"{'='*50}")
    print(f" LOW-MASS ALPHA:  {alpha_low:.4f}")
    print(f" HIGH-MASS ALPHA: {alpha_high:.4f}")
    print("-" * 50)
    print(f" ISP SIGNAL SAMPLES: {isp_count}")
    print(f" STD GRAVITY SAMPLES: {standard_count}")
    print("-" * 50)

    if alpha_high > 0.666:
        print(" RESULT: ISP GROWTH DETECTED.")
        print(f" Alpha increases by: {alpha_high - alpha_low:.4f}")
    else:
        print(" RESULT: NO GROWTH DETECTED.")
    print(f"{'='*50}\n")

    # 4. VISUALIZATION
    print(f"[4/4] RENDERING PROOF PLOTS...")
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True, facecolor='#121212')

    # Plot 1: Temperature vs Mass
    ax1.set_facecolor('#1e272e')
    ax1.scatter(M, T, c=T, cmap='plasma', s=20, alpha=0.4, label=r'Observed $T_{\text{obs}}$')
    ax1.set_xscale('log'); ax1.set_yscale('log')
    ax1.set_ylabel('Temperature (keV)', color='white')
    ax1.legend(loc='upper left'); ax1.grid(alpha=0.2)
    ax1.set_title('Mass-Temperature Relation', color='white')

    # Plot 2: The Exponent (The Proof)
    ax2.set_facecolor('#1e272e')
    # Color points by whether they are above or below the 2/3 threshold
    colors = np.where(local_alphas > 0.6666, 'lime', 'red')
    ax2.scatter(M, local_alphas, c=colors, s=25, alpha=0.6, label=r'Local Exponent ($\alpha$)')
    
    # Add the theoretical 2/3 line
    ax2.axhline(0.6666, color='white', ls=':', label=r'Standard Gravity (2/3)')
    
    # Trend line for the alpha
    log_M = np.log10(M)
    z = np.polyfit(log_M, local_alphas, 1)
    p = np.poly1d(z)
    ax2.plot(M, p(log_M), color='yellow', ls='--', label='Alpha Trend')

    ax2.set_xscale('log')
    ax2.set_ylabel(r'Exponent $\alpha = \frac{d(\log T)}{d(\log M)}$', color='white')
    ax2.set_xlabel('Mass ($M_{\odot}$)', color='white')
    ax2.legend(loc='upper left'); ax2.grid(alpha=0.2)
    ax2.set_title('The Smoking Gun: Scaling Exponent vs Mass', color='white')

    for ax in [ax1, ax2]:
        ax.tick_params(colors='white')
        for spine in ax.spines.values(): spine.set_edgecolor('white')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISP Local Slope Detector")
    parser.add_argument("file_a", help="Mass FITS file")
    parser.add_argument("file_b", help="Temp FITS file")
    parser.add_argument("--mass_col", default="SERSIC_MASS", help="Mass column")
    parser.add_argument("--temp_col", default="EP_HR1", help="Temp column (HR)")
    parser.add_argument("--radius", type=float, default=1.0, help="Match radius arcsec")
    
    args = parser.parse_args()
    main(args.file_a, args.file_b, args.mass_col, args.temp_col, args.radius)
