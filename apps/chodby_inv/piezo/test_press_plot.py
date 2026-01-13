import numpy as np
import matplotlib.pyplot as plt

def boxplot_opts(color: str, *, fill_only: bool = False) -> dict:
    """All ax.boxplot options in one dict.

    - Notches + median are fully opaque.
    - Box fill is color with alpha=0.2.
    - If fill_only=True, box boundary is removed (edgecolor='none').
    """
    alpha_box = 0.2
    edge = "none" if fill_only else color
    return dict(
        # geometry / stats
        vert=False,
        widths=0.08,
        whis=1.5,
        showfliers=False,

        # style
        patch_artist=True,
        boxprops=dict(facecolor=color, edgecolor=edge, linewidth=1.1, alpha=alpha_box),
        medianprops=dict(color=color, linewidth=3.0, alpha=1.0),
        whiskerprops=dict(color=color, linewidth=1.1, alpha=1.0),
        capprops=dict(color=color, linewidth=1.1, alpha=1.0),
    )

np.random.seed(7)

main_groups = ["Group A", "Group B", "Group C"]
n = 80

data_blue = [
    np.random.normal(0.0, 1.0, n),
    np.random.normal(0.8, 1.1, n),
    np.random.normal(1.5, 0.9, n),
]
data_red = [
    np.random.normal(0.3, 1.0, n),
    np.random.normal(1.1, 1.0, n),
    np.random.normal(1.9, 1.1, n),
]

base_pos = np.arange(1, len(main_groups) + 1)
offset = 0.18
pos_blue = base_pos - offset
pos_red  = base_pos + offset

fig, ax = plt.subplots(figsize=(8, 3.8))

# Choose one:
# fill_only=False  -> fill + boundary (both alpha 0.2)
# fill_only=True   -> fill only (no boundary)
fill_only = False

bp_blue = ax.boxplot(data_blue, positions=pos_blue, **boxplot_opts("tab:blue", fill_only=fill_only))
bp_red  = ax.boxplot(data_red,  positions=pos_red,  **boxplot_opts("tab:red",  fill_only=fill_only))

# Standard legend
bp_blue["boxes"][0].set_label("Blue subgroup")
bp_red["boxes"][0].set_label("Red subgroup")
ax.legend(loc="lower right", frameon=False)

ax.set_yticks(base_pos)
ax.set_yticklabels(main_groups)
ax.set_xlabel("Value")
ax.set_title("Notched thin boxes; median opaque; fill+edge alpha=0.2")
ax.grid(axis="x", alpha=0.25)

plt.tight_layout()
plt.show()
