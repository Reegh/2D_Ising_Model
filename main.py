from Functions.Simulation import run_temperature_sweep
from Functions.Graphics import plot_results

if __name__ == "__main__":
    # Optimized parameters for a good speed-quality balance
    results = run_temperature_sweep(
        L=50,              # Balanced size
        T_min=1.0,         # Heat rating
        T_max=3.5,
        n_T=25,            # Temperature points
        n_steps=10000      # Total steps (2000 balancing)
    )
    
    plot_results(results)
    print("Simulation completed! Results saved in Results")