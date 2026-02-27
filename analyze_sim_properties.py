import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import networkx as nx
from scipy.optimize import curve_fit

# Reuse loading logic
def load_sim_data(sim_data_path, sim_net_file):
    print(f"Loading sim network...")
    df = pd.read_csv(sim_net_file)
    G = nx.from_pandas_edgelist(df, 'user_1', 'user_2')
    num_nodes = G.number_of_nodes()
    
    npz_files = []
    if os.path.isfile(sim_data_path) and sim_data_path.endswith('.npz'):
        npz_files.append(sim_data_path)
    elif os.path.isdir(sim_data_path):
        for root, dirs, files in os.walk(sim_data_path):
            for file in files:
                if file.endswith(".npz"):
                    npz_files.append(os.path.join(root, file))
    
    if not npz_files:
        print(f"No .npz files found in {sim_data_path}")
        return None, num_nodes

    print(f"Found {len(npz_files)} simulation files. Loading and averaging...")
    
    all_active_counts = []
    all_cumulative_users = []

    for f in npz_files:
        data = np.load(f)
        X = data['X']
        # X shape: (steps, agents)
        
        # 1. Active counts per step (Volume/Impressions)
        active_counts = np.sum(X, axis=1)
        all_active_counts.append(active_counts)
        
        # 2. Cumulative Unique Users (Reach)
        # Check if user has ever been active up to step t
        # np.maximum.accumulate works on booleans as logical OR scan
        cum_mask = np.maximum.accumulate(X, axis=0)
        cumulative_users = np.sum(cum_mask, axis=1)
        all_cumulative_users.append(cumulative_users)

    # Align lengths
    min_len = min(len(a) for a in all_active_counts)
    aligned_counts = [a[:min_len] for a in all_active_counts]
    aligned_users = [a[:min_len] for a in all_cumulative_users]
    
    # Calculate mean
    avg_active_counts = np.mean(aligned_counts, axis=0)
    avg_cumulative_users = np.mean(aligned_users, axis=0)
    
    print(f"Averaged over {len(aligned_counts)} runs (min_len={min_len}).")
    
    return avg_active_counts, avg_cumulative_users, num_nodes

def logistic_model(t, K, r, t0):
    return K / (1 + np.exp(-r * (t - t0)))

def gompertz_model(t, K, r, t0):
    return K * np.exp(-np.exp(-r * (t - t0)))

def fit_growth_models(t, y_data, name="Data"):
    results = {}
    peak_idx_est = len(y_data) // 2 # Rough guess if not derived from rate
    
    # Logistic
    try:
        p0 = [y_data[-1], 0.5, peak_idx_est]
        popt, _ = curve_fit(logistic_model, t, y_data, p0=p0, maxfev=10000)
        residuals = y_data - logistic_model(t, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_data - np.mean(y_data))**2)
        r2 = 1 - (ss_res / ss_tot)
        print(f"   [{name}] Logistic Fit: R^2 = {r2:.4f} (K={popt[0]:.1f})")
        results['logistic'] = (popt, r2)
    except:
        results['logistic'] = (None, 0)

    # Gompertz
    try:
        p0 = [y_data[-1], 0.5, peak_idx_est]
        popt, _ = curve_fit(gompertz_model, t, y_data, p0=p0, maxfev=10000)
        residuals = y_data - gompertz_model(t, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_data - np.mean(y_data))**2)
        r2 = 1 - (ss_res / ss_tot)
        print(f"   [{name}] Gompertz Fit: R^2 = {r2:.4f} (K={popt[0]:.1f})")
        results['gompertz'] = (popt, r2)
    except:
        results['gompertz'] = (None, 0)
        
    return results

def analyze_properties(sim_activity, sim_cum_users):
    # sim_activity: instantaneous active counts (Volume per step)
    # sim_cum_users: cumulative unique users (Reach)
    
    cum_volume = np.cumsum(sim_activity) # Total Volume (Impressions)
    T = len(sim_activity)
    t = np.arange(T)
    
    print("\n--- Propagation Law Analysis ---")
    
    # 1. Peak Analysis (Daily Activity)
    peak_idx = np.argmax(sim_activity)
    peak_val = sim_activity[peak_idx]
    print(f"1. Peak Activity: {peak_val:.2f} agents at Step {peak_idx}")
    
    # 2. S-Curve Analysis
    print("2. S-Curve Growth Analysis:")
    results_volume = fit_growth_models(t, cum_volume, "Volume/Impressions")
    results_reach = fit_growth_models(t, sim_cum_users, "Reach/Unique Users")

    results = {}

    # 3. Decay Analysis (Power Law vs Exponential)
    # Analyze from peak onwards
    if peak_idx < T - 5: # Need enough points
        t_decay = t[peak_idx+1:]
        y_decay = sim_activity[peak_idx+1:]
        # Shift t to start from 1 for power law
        t_decay_shifted = t_decay - peak_idx 
        
        # Power Law: y = a * t^(-b) -> log(y) = log(a) - b * log(t)
        # Filter positive y only for log
        valid_mask = y_decay > 0
        if np.sum(valid_mask) > 5:
            y_log = np.log(y_decay[valid_mask])
            t_log = np.log(t_decay_shifted[valid_mask])
            
            # Linear regression for Power Law
            coeffs_pow = np.polyfit(t_log, y_log, 1)
            b_pow = -coeffs_pow[0]
            a_pow = np.exp(coeffs_pow[1])
            y_pred_pow = a_pow * (t_decay_shifted[valid_mask] ** -b_pow)
            r2_pow = 1 - np.sum((y_decay[valid_mask] - y_pred_pow)**2) / np.sum((y_decay[valid_mask] - np.mean(y_decay[valid_mask]))**2)
            
            print(f"4. Decay Analysis (Post-Peak):")
            print(f"   Power Law Decay: R^2 = {r2_pow:.4f}, alpha = {b_pow:.4f}")
            
            # Exponential: y = a * exp(-b * t) -> log(y) = log(a) - b * t
            coeffs_exp = np.polyfit(t_decay_shifted[valid_mask], y_log, 1)
            b_exp = -coeffs_exp[0]
            a_exp = np.exp(coeffs_exp[1])
            y_pred_exp = a_exp * np.exp(-b_exp * t_decay_shifted[valid_mask])
            r2_exp = 1 - np.sum((y_decay[valid_mask] - y_pred_exp)**2) / np.sum((y_decay[valid_mask] - np.mean(y_decay[valid_mask]))**2)
            
            print(f"   Exponential Decay: R^2 = {r2_exp:.4f}, lambda = {b_exp:.4f}")
            
            if r2_pow > r2_exp:
                print("   -> Decay follows Power Law better.")
            else:
                print("   -> Decay follows Exponential better.")
            
            results['decay'] = {'pow': (t_decay, a_pow * (t_decay_shifted ** -b_pow), r2_pow),
                                'exp': (t_decay, a_exp * np.exp(-b_exp * t_decay_shifted), r2_exp)}
        else:
             results['decay'] = None
    else:
        results['decay'] = None
        
    # Plotting
    plt.figure(figsize=(16, 6))
    
    # 1. Daily Activity & Decay
    plt.subplot(1, 3, 1)
    plt.plot(t, sim_activity, 'b-', label='Daily Active (Volume)')
    plt.axvline(x=peak_idx, color='gray', linestyle='--', label=f'Peak ({peak_idx})')
    
    if results.get('decay'):
        t_decay = results['decay']['pow'][0]
        y_pow = results['decay']['pow'][1]
        y_exp = results['decay']['exp'][1]
        r2_pow = results['decay']['pow'][2]
        r2_exp = results['decay']['exp'][2]
        
        plt.plot(t_decay, y_pow, 'g--', label=f'Power Law (R2={r2_pow:.2f})')
        plt.plot(t_decay, y_exp, 'r:', label=f'Exp Decay (R2={r2_exp:.2f})')

    plt.title('Daily Activity (Decay)')
    plt.xlabel('Step')
    plt.ylabel('Active Count')
    plt.legend()
    
    # 2. Cumulative Users (Reach)
    plt.subplot(1, 3, 2)
    plt.plot(t, sim_cum_users, 'k-', label='Unique Users (Reach)', linewidth=2)
    
    if results_reach['logistic'][0] is not None:
        popt, r2 = results_reach['logistic']
        plt.plot(t, logistic_model(t, *popt), 'r--', label=f'Logistic (R2={r2:.2f})')
        
    if results_reach['gompertz'][0] is not None:
        popt, r2 = results_reach['gompertz']
        plt.plot(t, gompertz_model(t, *popt), 'g:', label=f'Gompertz (R2={r2:.2f})')
        
    plt.title('Cumulative Unique Users (Reach)')
    plt.xlabel('Step')
    plt.ylabel('Total Users')
    plt.legend()

    # 3. Cumulative Volume (Impressions)
    plt.subplot(1, 3, 3)
    plt.plot(t, cum_volume, 'm-', label='Total Volume (Impressions)', linewidth=2)
    
    if results_volume['logistic'][0] is not None:
        popt, r2 = results_volume['logistic']
        plt.plot(t, logistic_model(t, *popt), 'r--', label=f'Logistic (R2={r2:.2f})')
        
    if results_volume['gompertz'][0] is not None:
        popt, r2 = results_volume['gompertz']
        plt.plot(t, gompertz_model(t, *popt), 'g:', label=f'Gompertz (R2={r2:.2f})')

    plt.title('Cumulative Total Volume')
    plt.xlabel('Step')
    plt.ylabel('Total Impressions')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('propagation_analysis_result.png')
    print("Analysis plot saved to propagation_analysis_result.png")

def main():
    sim_net_file = '/mlp_vepfs/share/zsz/simulation_agent/data/relationship_1000.csv'
    sim_file_path = '/mlp_vepfs/share/zsz/simulation_agent/simulation_data/Toy_Story/temp/'
    
    sim_activity, sim_cum_users, _ = load_sim_data(sim_file_path, sim_net_file)
    if sim_activity is not None:
        analyze_properties(sim_activity, sim_cum_users)

if __name__ == "__main__":
    main()
