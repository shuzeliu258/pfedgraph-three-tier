import numpy as np
import matplotlib.pyplot as plt

files = [
    "results_global_acc_run_1765447501.npy",
    "results_global_acc_run_1765449476.npy",
    "results_global_acc_run_1765451478.npy",
    "results_global_acc_run_1765453485.npy"
]

acc_all = np.array([np.load(f) for f in files])    
acc_all = acc_all.astype(float)                    

sample_interval = 5
acc_sampled = acc_all[:, ::sample_interval]        
rounds = np.arange(acc_all.shape[1])[::sample_interval]

mean_acc = np.nanmean(acc_sampled, axis=0)
std_acc  = np.nanstd(acc_sampled, axis=0)

plt.figure(figsize=(8,5))

plt.plot(rounds, mean_acc, color='blue', linewidth=2, label='Mean Accuracy')
plt.fill_between(
    rounds,
    mean_acc - std_acc,
    mean_acc + std_acc,
    color='blue',
    alpha=0.2,
    label='±1 Std'
)

plt.xlabel("Communication Round")
plt.ylabel("Global Accuracy")
plt.title("Mean ± Std Accuracy (Sampled every 5 rounds)")

plt.ylim(0, 1)

plt.grid(True, linestyle='--', alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig("global_acc_mean_std.png", dpi=300)
print("Saved figure to global_acc_mean_std.png")

plt.show()
