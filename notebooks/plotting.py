import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_training_loss_history(loss_hist_train, startidx=0, 
                               log=False, title="Training Loss", 
                               saveas=None):
    fig, ax = plt.subplots(1, 1)
    erange = np.arange(startidx, len(loss_hist_train))
    fplot = ax.semilogy if log else ax.plot
    fplot(1 + erange, loss_hist_train[erange], '.-')
    ax.set_xlabel(f"epoch")
    ax.set_ylabel(f"loss")
    ax.set_title(title)
    if saveas: plt.savefig(saveas)
    return ax

def plot_validation_loss_history(loss_hist_valid, startidx=0, 
                                 log=False, title="Validation Loss", 
                                 saveas=None):
    fig, ax = plt.subplots(1, 1)
    erange = np.arange(startidx, len(loss_hist_valid))
    fplot = ax.semilogy if log else ax.plot
    fplot(1 + erange, loss_hist_valid[erange], '.-')
    ax.set_xlabel(f"epoch")
    ax.set_ylabel(f"loss")
    ax.set_title(title)
    if saveas: plt.savefig(saveas)
    return ax

def plot_landscape(
        phi_func, 
        r=2, 
        res=100, 
        plot3d=False,
        signal=[0,0], 
        lognormalize=True, 
        cmap='coolwarm', 
        xlims=None, 
        ylims=None, 
        xlabel="$x$", 
        ylabel="$y$", 
        zlabel="$z$",
        title="$\phi$",
        cbar_title="$\phi$", 
        ax=None,
        figsize=(6, 4),
        view_init=(30, -45),
        clip=None,
        saveas=None,
    ):

    if ax is None and plot3d:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    elif ax is None and (not plot3d):
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    signal_params = np.array([[10, *signal, *signal]])
    signal_params = torch.tensor(signal_params, dtype=torch.float32)

    x = np.linspace(-r, r, res)
    y = np.linspace(-r, r, res)
    xs, ys = np.meshgrid(x, y)
    z = np.array([xs.flatten(), ys.flatten()]).T
    z = torch.tensor(z, dtype=torch.float32, requires_grad=True)[None,:]

    phi_star = phi_func(xs, ys, signal[0], signal[1])
    if lognormalize:
        phi_star = np.log(1 + phi_star - phi_star.min())  # log normalize

    clip = phi_star.max() + 1 if clip is None else clip
    under_cutoff = phi_star <= clip
    plot_screen = np.ones(under_cutoff.shape)
    plot_screen[~under_cutoff] = np.nan
    phi_star_plot = phi_star * plot_screen

    # Plot
    if plot3d:
        sc = ax.plot_surface(
            xs, ys, phi_star_plot.reshape(xs.shape),
            vmin=phi_star[under_cutoff].min(),
            vmax=phi_star[under_cutoff].max(),
            cmap=cmap
        )
    else:
        sc = ax.pcolormesh(
            xs, ys, phi_star_plot.reshape(xs.shape),
            vmin=phi_star[under_cutoff].min(),
            vmax=phi_star[under_cutoff].max(),
            cmap=cmap, 
        )

    cbar = plt.colorbar(sc)
    cbar.ax.set_title(cbar_title, size=10)
    cbar.ax.tick_params(labelsize=8)

    # Format plot
    if xlims is not None: ax.set_xlim(*xlims)
    if ylims is not None: ax.set_ylim(*ylims)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if plot3d: 
        ax.set_zlabel(zlabel)
        ax.view_init(*view_init)
    plt.tight_layout()
    
    if saveas: plt.savefig(saveas)
    
    return ax
