import matplotlib.pyplot as plt
import numpy as np


def arranged_img(sample) -> None:
    image = sample["image"].numpy()
    rss = sample["rss_from_coils"].numpy()
    target = sample["target_rss"].numpy()
    valid = sample["coil_combine_valid"].numpy()

    vmax = max(float(rss.max()), float(np.abs(image).max()),
            float(target.max()), 1e-12)

    fig, axes = plt.subplots(2, 5, figsize=(17, 7))

    for i in range(2):
        phase = np.ma.masked_where(
            (~valid[i]) | (np.abs(image[i]) <= 0.01 * np.abs(image[i]).max()),
            np.angle(image[i]),
        )
        panels = [
            ("Coil RSS", rss[i], "gray", 0, vmax),
            ("Combined magnitude", np.abs(image[i]), "gray", 0, vmax),
            ("Stored/fallback target", target[i], "gray", 0, vmax),
            ("Combined phase", phase, "twilight", -np.pi, np.pi),
            ("Valid coil support", valid[i], "gray", 0, 1),
        ]
        for ax, (title, value, cmap, lo, hi) in zip(axes[i], panels):
            ax.imshow(value, cmap=cmap, vmin=lo, vmax=hi)
            ax.set_title(title)
            ax.axis("off")

    plt.tight_layout()
    plt.show()