def plot_metrics(rows, path):
    """Plot actual attempt placement errors without inferring success."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6,3))
    ax.plot([r["placement_error_m"] for r in rows],"o")
    ax.set(xlabel="Executed attempt",ylabel="Final XY placement error (m)")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
