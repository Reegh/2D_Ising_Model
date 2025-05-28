from Functions.Simulation import run_temperature_sweep
from Functions.Graphics import plot_results

if __name__ == "__main__":
    # Parámetros optimizados para buen balance velocidad-calidad
    results = run_temperature_sweep(
        L=50,              # Tamaño balanceado
        T_min=1.0,         # Rango térmico
        T_max=3.5,
        n_T=25,            # Puntos de temperatura
        n_steps=10000      # Pasos totales (2000 de equilibración)
    )
    
    plot_results(results)
    print("¡Simulación completada! Resultados guardados en Results")