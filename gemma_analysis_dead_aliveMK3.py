import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys
import os

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
H_ALPHA_THRESHOLD = 3.0 
STANDARD_ALPHA = 0.6667

# ---------------------------------------------------------
# CORE MATH ENGINE
# ---------------------------------------------------------

def run_regression(df, label):
    """Performs log-log regression to find alpha and mean Delta."""
    if df is None or len(df) < 3:
        return {
            "label": label,
            "count": len(df) if df is not None else 0,
            "is_empty": True
        }

    df = df.copy()
    # Standard Model: T_pred = M^(2/3)
    df['T_pred'] = df['Mass']**(2/3)
    df['Delta'] = df['Temp_Obs'] / df['T_pred']

    log_m = np.log10(df['Mass'])
    log_t = np.log10(df['Temp_Obs'])
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_m, log_t)
    
    return {
        "label": label,
        "count": len(df),
        "is_empty": False,
        "alpha": slope,
        "delta_avg": df['Delta'].mean(),
        "r_squared": r_value**2,
        "p_value": p_value,
        "df": df
    }

# ---------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python isp_unified_test.py <file1.csv> <file2.csv> ...")
        return

    files_to_process = sys.argv[1:]
    all_data_frames = []

    print(f"[*] Starting Unified Batch Process for {len(files_to_process)} files.")

    # 1. AGGREGATE ALL DATA
    for filename in files_to_process:
        try:
            temp_df = pd.read_csv(filename)
            # Ensure core columns exist
            if 'Mass' in temp_df.columns and 'Temp_Obs' in temp_df.columns:
                all_data_frames.append(temp_df)
                print(f"    [+] Loaded {filename}")
            else:
                print(f"    [-] SKIPPING {filename}: Missing Mass or Temp_Obs")
        except Exception as e:
            print(f"    [-] ERROR loading {filename}: {e}")

    if not all_data_frames:
        print("[-] ERROR: No valid data loaded from any files.")
        return

    # Combine everything into one master dataframe
    master_df = pd.concat(all_data_frames, ignore_index=True)
    master_df = master_df.dropna(subset=['Mass', 'Temp_Obs'])

    # 2. THE UNIFIED SPLIT
    print("\n" + "="*60)
    print("          UNIFIED ISP PHASE TRANSITION TEST          ")
    print("="*60)
    print("Categorizing all data into Live vs Dead populations...")

    # LOGIC:
    # If Halpha_EW is missing (NaN) -> It is LIVE (baseline)
    # If Halpha_EW > threshold      -> It is LIVE
    # If Halpha_EW <= threshold     -> It is DEAD
    if 'Halpha_EW' in master_df.columns:
        live_mask = (master_df['Halpha_EW'].isna()) | (master_df['Halpha_EW'] > H_ALPHA_THRESHOLD)
        dead_mask = (master_df['Halpha_EW'] <= H_ALPHA_THRESHOLD)
    else:
        print("[!] WARNING: No H-alpha column found in ANY file.")
        print("    Treating all data as one single group (Baseline).")
        live_mask = np.ones(len(master_df), dtype=bool)
        dead_mask = np.zeros(len(master_df), dtype=bool)

    live_df = master_df[live_mask].copy()
    dead_df = master_df[dead_mask].copy()

    print(f"  Total Objects Analyzed: {len(master_df)}")
    print(f"  [+] LIVE Population:    {len(live_df)}")
    print(f"  [+] DEAD Population:    {len(dead_df)}")
    print("-" * 60)

    # 3. RUN REGRESSIONS
    results = []
    results.append(run_regression(live_df, "LIVE (Hydrogen Present/Baseline)"))
    results.append(run_regression(dead_df, "DEAD (No Hydrogen)"))

    # 4. REPORTING
    print("\n" + "="*60)
    print("                 STATISTICAL RESULTS                          ")
    print("==============================================================")

    for res in results:
        if res["is_empty"]:
            print(f"\n[!] Group '{res['label']}' is too small/empty to analyze.")
            continue

        print(f"\nGROUP: {res['label']}")
        print(f"  Samples:           {res['count']}")
        print(f"  Measured Alpha:    {res['alpha']:.4f} (Std: {STANDARD_ALPHA})")
        print(f"  Mean Delta (Δ):    {res['delta_avg']:.4f} (Std: 1.0)")
        print(f"  R-squared:         {res['r_squared']:.4f}")
        print(f"  P-value:           {res['p_value']:.4e}")

        if res['p_value'] < 0.05:
            print("  VERDICT: SIGNIFICANT deviation from Standard Model.")
            if res['delta_avg'] > 1.1:
                print("  SIGNAL: Positive Heat Kick detected.")
            else:
                print("  SIGNAL: No Heat Kick (Standard/Negative).")
        else:
            print("  VERDICT: NOT SIGNIFICANT (Follows Standard Model).")
    print("="*60)

    # 5. HYPOTHESIS TEST
    if not results[0]['is_empty'] and not results[1]['is_empty']:
        print("\n[HYPOTHESIS TEST: LIVE vs DEAD]")
        diff = results[1]['delta_avg'] - results[0]['delta_avg']
        print(f"Delta (Live): {results[0]['delta_avg']:.4f}")
        print(f"Delta (Dead): {results[1]['delta_avg']:.4f}")
        print(f"Difference (Δ_dead - Δ_live): {diff:.4f}")

        if diff > 0.3:
            print("\n[!!!] SUCCESS: HYPOTHESIS SUPPORTED [!!!]")
            print("The 'Heat Kick' is significantly higher in the DEAD group.")
            print("This matches the theory of an ISP matter-dumping event.")
        elif diff < -0.3:
            print("\n[?] HYPOTHESIS CONTRADICTED: Dead galaxies are cooler.")
        else:
            print("\n[?] NO CLEAR SIGNAL: The difference is not statistically significant.")
    print("="*60 + "\n")

    # 6. VISUALIZATION
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for res in results:
        if not res['is_empty']:
            ax1.scatter(res['df']['Mass'], res['df']['Temp_Obs'], label=res['label'], alpha=0.6)
            ax2.scatter(res['df']['Mass'], res['df']['Delta'], label=res['label'], alpha=0.6)

    m_range = np.linspace(master_df['Mass'].min(), master_df['Mass'].max(), 100)
    ax1.plot(m_range, m_range**(2/3), color='black', linestyle='--', label='Standard Model')
    ax2.axhline(y=1, color='black', linestyle='--')

    ax1.set_xscale('log'); ax1.set_yscale('log')
    ax1.set_xlabel(r'Mass ($10^{14} M_{\odot}$)')
    ax1.set_ylabel(r'Temperature (keV)')
    ax1.set_title('Mass-Temperature Scaling')
    ax1.legend(); ax1.grid(True, alpha=0.2)

    ax2.set_xscale('log')
    ax2.set_xlabel(r'Mass ($10^{14} M_{\odot}$)')
    ax2.set_ylabel(r'$\Delta$ (Observed / Predicted)')
    ax2.set_title('The "Heat Kick" Residuals')
    ax2.legend(); ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    output_plot = "ISP_Unified_Comparison_Result.png"
    plt.savefig(output_plot)
    print(f"[!] Plot saved as '{output_plot}'")
    plt.show()

if __name__ == "__main__":
    main()

