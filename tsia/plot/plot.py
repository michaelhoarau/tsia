def plot_mtf_metrics(mtf, tag):
    fig = plt.figure(figsize=(26.5, 4))
    gs = gridspec.GridSpec(nrows=1, ncols=3, width_ratios=[1, 1, 1])

    ax = fig.add_subplot(gs[0])
    ax.hist(np.diag(mtf), color=colors[1], edgecolor='#FFFFFF')
    ax.set_title('Distribution of self-transition probabilities')

    ax = fig.add_subplot(gs[1])
    means, stds = [], []
    for diag_index in range(0, image_size):
        diag = np.diag(mtf, k=diag_index)
        means.append(diag.mean())
        stds.append(diag.std())

        diag = np.hstack((np.zeros(shape=(diag_index,)), diag))
        ax.plot(diag, linewidth=1.5 - diag_index/10, alpha=1 - diag_index/10)
    ax.set_title('Transition probabilities across the MTF diagonals')

    ax = fig.add_subplot(gs[2])
    ax.plot(means, label='MTF diagonals mean')
    ax.plot(stds, label='MTF diagonals std')
    ax.set_title('Probability means and standard deviation across the MTF diagonals')
    ax.legend()

    plt.show()

def plot_quantile(timeseries, tag, bin_edges):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    alpha = 0.1
    fig, axes = utils.plot_timeseries(timeseries, tag, custom_grid=False, fig_height=8, fig_width=28)

    for index in range(len(bin_edges) - 1):
        axes[0].fill_between(timeseries.index, y1=bin_edges[index], y2=bin_edges[index+1], alpha=alpha, color=colors[(index + 1) % len(colors)])

def plot_network_graph(mtf, ax, title, cmap='jet'):
    G = nx.from_numpy_matrix(mtf)
    colormap = plt.cm.get_cmap(cmap).reversed()

    partitions = community.best_partition(G)
    nb_partitions = len(set(partitions.values()))
    partition = community.best_partition(G)
    modularity = community.modularity(communities, G)

    # Compute node colors and edges colors for the modularity encoding:
    edge_colors = [matplotlib.colors.to_hex(cm.get_cmap(cmap)(communities.get(v)/(nb_partitions - 1))) for u,v in G.edges()]
    node_colors = [communities.get(node) for node in G.nodes()]
    node_size = list(nx.clustering(G, weight='weight').values())
    node_size = list((node_size - np.min(node_size)) * 2000 + 10)

    # Builds the options set to draw the network graph in the "modularity" configuration:
    options = {
        'node_size': node_size,
        'edge_color': edge_colors,
        'node_color': node_colors,
        'edgecolors': '#000000',    # Color of the node edges
        'linewidths': 1,            # Width of the node edges
        'width': 0.1,               # Width of the edges
        'alpha': 0.6,               # Transparency of both edges and nodes
        'with_labels': False,
        'cmap': cmap
    }

    nx.draw_networkx(G, **options, pos=nx.spring_layout(G), ax=ax)
    ax.set_title(rf'Partitions: $\bf{nb_partitions}$ - Modularity: $\bf{modularity:.3f}$', fontsize=10)
    ax.axis("off")
    
    return nb_partitions, modularity, options