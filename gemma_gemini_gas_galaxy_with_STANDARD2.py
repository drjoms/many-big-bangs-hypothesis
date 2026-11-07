import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import minimize

# Versioning
__version__ = "6.3.5-LIMIT_ANALYSIS"

# ==============================================================================
# HARDCODED VOID GAS ANCHORS (Source: Cooke et al. 2011, MNRAS 417, 1534)
# ==============================================================================
VOID_ANCHORS_DATA = [
    [1e7, 6.33e-6, "DLA_J0035"],
    [1e7, 5.30e-5, "DLA_J0831"],
    [1e7, 1.94e-5, "DLA_Q0913"],
    [1e7, 5.34e-6, "DLA_J1001"],
    [1e7, 5.84e-5, "DLA_J1037"],
    [1e7, 1.33e-3, "DLA_Q1108"],
    [1e7, 2.51e-5, "DLA_J1337"],
    [1e7, 1.22e-3, "DLA_J1558"],
    [1e7, 4.23e-5, "DLA_Q1946"],
    [1e7, 1.33e-3, "DLA_Q2059"],
    [1e7, 7.32e-5, "DLA_J2155"],
    [1e7, 2.83e-5, "DLA_Q2206"],
    [1e7, 1.61e-6, "Star_CS22949"],
    [1e7, 2.52e-5, "Star_HD122563"],
]

def power_law_sigmoid(M, L, k, M_crit, floor):
    """Numerically stable log-space sigmoid."""
    try:
        log_ratio = np.log(M) - np.log(M_crit)
        exponent = -k * log_ratio
        exponent = np.clip(exponent, -700, 700)
        return floor + (L / (1 + np.exp(exponent)))
    except:
        return floor

def weighted_loss_function(params, M, y, weights):
    """Weighted Least Squares loss."""
    L, k, M_crit, floor = params
    if L <= 0.0 or k <= 0.0 or M_crit <= 0.0 or floor < 0.0 or floor > 0.1:
        return 1e16
    preds = power_law_sigmoid(M, L, k, M_crit, floor)
    if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
        return 1e16
    return np.sum(weights * (y - preds) ** 2)

def main():
    parser = argparse.ArgumentParser(description="MZR Analysis - Divergence & Limit Analysis")
    parser.add_argument("fits_path", help="Path to the massive FITS catalog")
    parser.add_argument("--mass_col", default="SERSIC_MASS", help="Column for Mass")
    parser.add_argument("--met_col", default="Z", help="Column for Metallicity")
    parser.add_argument("--weight", type=float, default=500.0, help="Weight multiplier for Anchors")
    parser.add_argument("--sigma", type=float, default=3.5, help="Sigma threshold for clipping 'BS' data")
    args = parser.parse_args()

    print(f"\n==================================================", flush=True)
    print(f" INITIALIZING LIMIT ANALYSIS ENGINE v{__version__}", flush=True)
    print(f"==================================================\n", flush=True)

    # 1. RAM INGESTION
    print(f"[1/5] Ingesting {args.fits_path}...", flush=True)
    try:
        with fits.open(args.fits_path, memmap=False) as hdul:
            data = hdul[1].data
            df = pd.DataFrame({
                'mass': data[args.mass_col].ravel().astype(float),
                'met': data[args.met_col].ravel().astype(float),
                'weight': 1.0,
                'is_anchor': False
            })
        print(f"   -> Success: {len(df)} rows loaded into RAM.", flush=True)
    except Exception as e:
        print(f" [!] FITS Error: {e}", flush=True)
        sys.exit(1)

    # 2. CLEANING & ANCHOR INJECTION
    print(f"[2/5] Injecting Anchors (Weight: {args.weight}x) and Cleaning...", flush=True)
    df = df[(df['mass'] > 0) & (df['met'] > 1e-10)].copy()
    anchors_df = pd.DataFrame(VOID_ANCHORS_DATA, columns=['mass', 'met', 'label'])
    anchors_df['is_anchor'] = True
    anchors_df['weight'] = args.weight
    combined_df = pd.concat([df[['mass', 'met', 'weight', 'is_anchor']], 
                             anchors_df[['mass', 'met', 'weight', 'is_anchor']]], 
                            ignore_index=True)
    
    # 3. FIRST PASS: DISCOVERY FIT
    print(f"[3/5] Running Discovery Fit (Initial Pass)...", flush=True)
    M = combined_df['mass'].values
    y = combined_df['met'].values
    w = combined_df['weight'].values
    M_guess = np.median(M)
    p0 = [y.max() * 0.8, 1.0, M_guess, y.min()]
    result = minimize(weighted_loss_function, p0, args=(M, y, w), method='Nelder-Mead', options={'maxiter': 15000})
    
    # 4. SIGMA CLIPPING
    print(f"[4/5] Identifying Low-Fidelity Survey Data (> {args.sigma}$\sigma$)...", flush=True)
    y_fit_initial = power_law_sigmoid(M, *result.x)
    residuals = np.abs(y - y_fit_initial)
    survey_mask = combined_df['is_anchor'] == False
    res_std = residuals[survey_mask].std()
    outlier_mask = (residuals > (args.sigma * res_std)) & (combined_df['is_anchor'] == False)
    cleaned_df = combined_df[~outlier_mask].copy()

    # 5. FINAL PASS: REFINED ISP FIT & STANDARD MODEL CALCULATION
    print(f"[5/5] Calculating Refined ISP Fit vs. Standard Model...", flush=True)
    M_c = cleaned_df['mass'].values
    y_c = cleaned_df['met'].values
    w_c = cleaned_df['weight'].values

    # ISP Model (The Sigmoid)
    result_final = minimize(weighted_loss_function, result.x, args=(M_c, y_c, w_c), method='Nelder-Mead', options={'maxiter': 15000})
    L, k, M_crit, floor = result_final.x
    y_fit_isp = power_law_sigmoid(M_c, L, k, M_crit, floor)
    isp_saturation = L + floor

    # Standard Model (Linear Regression in Log-Log space: log(Z) = A + b*log(M))
    survey_only = cleaned_df[cleaned_df['is_anchor'] == False]
    log_m_std = np.log10(survey_only['mass'].values)
    log_z_std = np.log10(survey_only['met'].values)
    slope_std, intercept_std = np.polyfit(log_m_std, log_z_std, 1)
    
    # --- CALCULATE DIVERGENCE POINT ---
    # Solve for M where: 10^(intercept + slope * log10(M)) = isp_saturation
    # log10(M) = (log10(isp_saturation) - intercept) / slope
    log_m_divergence = (np.log10(isp_saturation) - intercept_std) / slope_std
    M_divergence = 10**log_m_divergence

    # Stats
    res_isp = y_c - y_fit_isp
    r_sq_isp = 1 - (np.sum(res_isp**2) / np.sum((y_c - np.mean(y_c))**2))

    # DATA DUMP
    print("\n" + "="*50, flush=True)
    print(f"SCIENTIFIC LIMIT ANALYSIS - {__version__}", flush=True)
    print("="*50, flush=True)
    print("ISP MODEL (CONVERGENT):", flush=True)
    print(f"Saturation Ceiling: {isp_saturation:.8f}", flush=True)
    print(f"Critical Scale:     {M_crit:.4e} M_sun", flush=True)
    print("-" * 50, flush=True)
    print("STANDARD MODEL (DIVERGENT):", flush=True)
    print(f"Power-law Slope:    {slope_std:.8f}", flush=True)
    print(f"Unbounded Growth:   Yes (Z -> infinity)", flush=True)
    print("-" * 50, flush=True)
    print("LIMIT COMPARISON:", flush=True)
    print(f"ISP Physical Max:   {isp_saturation:.8f}", flush=True)
    print(f"Divergence Point:   {M_divergence:.4e} M_sun", flush=True)
    print(" (Mass where Standard Model exceeds ISP ceiling)", flush=True)
    print("-" * 50, flush=True)
    print(f"ISP R-squared:      {r_sq_isp:.8f}", flush=True)
    print("="*50 + "\n", flush=True)

    # VISUALIZATION
    print("[VISUALIZATION] Rendering Divergence Plot...", flush=True)
    plt.figure(figsize=(14, 8), facecolor='#121212')
    ax = plt.gca()
    ax.set_facecolor('#141414')

    # 1. Background (Cleaned Survey)
    plt.scatter(survey_only['mass'], survey_only['met'], c=survey_only['met'], 
                cmap='plasma', s=0.5, alpha=0.1, label='Survey Population', zorder=1)

    # 2. The Anchors
    anchors_only = cleaned_df[cleaned_df['is_anchor'] == True]
    plt.scatter(anchors_only['mass'], anchors_only['met'], color='lime', marker='s', s=70, 
                edgecolors='white', label='High-Fidelity Anchors (VLT/Keck)', zorder=4)

    # 3. The Standard Model (The "Failure" Line)
    # We extend the range significantly to show the divergence
    M_range = np.logspace(np.log10(M_c.min()), np.log10(M_c.max() * 20), 1000) 
    y_fit_std = 10**(intercept_std + slope_std * np.log10(M_range))
    plt.plot(M_range, y_fit_std, color='magenta', linestyle='--', linewidth=2, 
             label=f'Standard Model ($\log Z \propto {slope_std:.2f} \log M$)', zorder=2)

    # 4. The ISP Fit Line (The "Success" Line)
    y_fit_isp_plot = power_law_sigmoid(M_range, L, k, M_crit, floor)
    plt.plot(M_range, y_fit_isp_plot, color='cyan', linewidth=3, label='ISP Refined Fit (Sigmoid)', zorder=3)

    # 5. The Divergence Marker
    plt.axvline(M_divergence, color='red', linestyle=':', alpha=0.6, label='Standard Model Divergence Point', zorder=5)

    plt.xscale('log')
    plt.title(f"MZR: ISP Hypothesis vs Standard Model | {__version__}", color='white', fontsize=16)
    plt.xlabel(r"Stellar Mass ($M_{\odot}$)", color='white', fontsize=12)
    plt.ylabel("Gas Metallicity (Z)", color='white', fontsize=12)
    plt.tick_params(colors='white')
    plt.grid(True, which="both", color='#242424', linestyle=':', alpha=0.5)

    # Parameter Text Box
    text_box = (
        f"ISP MODEL (Convergent):\n"
        f"Saturation: {isp_saturation:.4f}\n"
        f"M_crit:     {M_crit:.2e}\n"
        f"R²:         {r_sq_isp:.4f}\n\n"
        f"STANDARD MODEL (Divergent):\n"
        f"slope:      {slope_std:.4f}\n"
        f"Divergence: {M_divergence:.2e}"
    )
    plt.gca().text(0.05, 0.95, text_box, transform=plt.gca().transAxes, fontsize=11,
                   color='white', fontfamily='monospace', verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='#1a1a1a', edgecolor='#444', alpha=0.8))

    plt.legend(facecolor='#1c1c1c', labelcolor='white', loc='lower right')
    plt.tight_layout()
    plt.show()
    
    print(f" 🏁 [FINISH] Pipeline v{__version__} complete.", flush=True)

if __name__ == "__main__":
    main()

