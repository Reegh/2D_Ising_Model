import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Style settings
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 12

def plot_results(results):
    """Professional visualization of results."""
    Ts = np.array([r['temperature'] for r in results])
    Es = np.array([r['avg_energy'] for r in results])
    Ms = np.array([r['abs_magnetization'] for r in results])
    Cs = np.array([r['heat_capacity'] for r in results])
    
    fig, ax = plt.subplots(3, 1, figsize=(10, 12))
    
    # Energy graph
    plt.figure(figsize=(8, 5))
    plt.plot(Ts, Es, 'o-', color='royalblue', label='Simulation')
    plt.axvline(x=2.269, color='red', linestyle='--', label='Critical T')
    plt.xlabel(r'Temperature')
    plt.ylabel(r'Energy')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('Results/ising_energy.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Magnetization graph
    plt.figure(figsize=(8, 5))
    plt.plot(Ts, Ms, 'o-', color='darkorange', label='Simulation')
    plt.axvline(x=2.269, color='red', linestyle='--')
    plt.xlabel(r'Temperature')
    plt.ylabel('Magnetization')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('Results/ising_magnetization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Heat capacity graph
    plt.figure(figsize=(8, 5))
    plt.plot(Ts, Cs, 'o-', color='forestgreen', label='Simulation')
    plt.axvline(x=2.269, color='red', linestyle='--')
    plt.xlabel(r'Temperature')
    plt.ylabel(r'Heat Capacity')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('Results/ising_heat_capacity.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot spin configurations for selected temperatures
    for result in results:
        if result['temperature'] in [min(Ts), 2.0, 2.269, max(Ts)]:  # Key temperatures
            plot_spin_configurations(
                initial_spins=result['initial_spins'],
                final_spins=result['final_spins'],
                temperature=result['temperature']
            )
            if result.get('frames') is not None:
                animate_spin_frames(result['frames'], result['temperature'])

def animate_spin_frames(frames, temperature):

    os.makedirs('Results', exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(frames[0], cmap='coolwarm', vmin=-1, vmax=1)
    ax.axis('off')
    ax.set_title(f'Spin Evolution (T = {temperature:.3f})')

    def update(frame):
        im.set_data(frame)
        return [im]

    ani = animation.FuncAnimation(
        fig, update, frames=frames, blit=True, interval=100
    )

    # Save as GIF
    #ani.save(f"Results/spin_animation_T_{temperature:.3f}.gif", writer='pillow')

    # Save as MP4 (requires ffmpeg)
    ani.save(f"Results/spin_animation_T_{temperature:.3f}.mp4", writer='ffmpeg', fps=10)

    plt.close()


def plot_spin_configurations(initial_spins, final_spins, temperature):
    """Visualize the initial and final configurations of the spins."""
    # Creates the figure with a manual design
    fig = plt.figure(figsize=(12, 6))
    
    # Adds subplots with manual positions
    ax1 = fig.add_axes([0.05, 0.1, 0.4, 0.8])  # [left, bottom, width, height]
    ax2 = fig.add_axes([0.5, 0.1, 0.4, 0.8])
    
    # Initial configuration
    im1 = ax1.imshow(initial_spins, cmap='coolwarm', vmin=-1, vmax=1)
    ax1.set_title(f'Initial Configuration (T = {temperature:.3f})')
    ax1.axis('off')
    
    # Final configuration
    im2 = ax2.imshow(final_spins, cmap='coolwarm', vmin=-1, vmax=1)
    ax2.set_title(f'Final Configuration (T = {temperature:.3f})')
    ax2.axis('off')
    
    # Shared color bar
    cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])  # Color bar position
    fig.colorbar(im2, cax=cax, label='Spin value')
    cax.set_yticks([-1, 0, 1])
    
    plt.savefig(f'Results/spin_configs_T_{temperature:.3f}.png', dpi=300, bbox_inches='tight')
    plt.close()