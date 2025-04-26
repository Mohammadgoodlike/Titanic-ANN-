import matplotlib.pyplot as plt
import networkx as nx



def plot_neural_network(input_size, hidden_layers, output_size):
    G = nx.Graph()


    for i in range(input_size):
        G.add_node(f'I{i + 1}', layer='Input', pos=(0, -i-1.5))


    for layer_idx, layer_size in enumerate(hidden_layers):
        for j in range(layer_size):
            G.add_node(f'H{layer_idx + 1}_{j + 1}', layer=f'Hidden Layer {layer_idx + 1}', pos=(layer_idx + 1, -j))


    for k in range(output_size):
        G.add_node(f'O{k + 1}', layer='Output', pos=(len(hidden_layers) + 1, -3.5))


    for i in range(input_size):
        for j in range(hidden_layers[0]):
            G.add_edge(f'I{i + 1}', f'H1_{j + 1}')


    for layer_idx in range(len(hidden_layers) - 1):
        for j in range(hidden_layers[layer_idx]):
            for k in range(hidden_layers[layer_idx + 1]):
                G.add_edge(f'H{layer_idx + 1}_{j + 1}', f'H{layer_idx + 2}_{k + 1}')


    for j in range(hidden_layers[-1]):
        for k in range(output_size):
            G.add_edge(f'H{len(hidden_layers)}_{j + 1}', f'O{k + 1}')


    pos = nx.get_node_attributes(G, 'pos')


    colors = ['#ff9999']
    colors += ['#66b3ff'] * len(hidden_layers)
    colors.append('#99ff99')
    layer_mapping = {'Input': colors[0], 'Output': colors[-1]}
    for i in range(1, len(hidden_layers) + 1):
        layer_mapping[f'Hidden Layer {i}'] = colors[i]


    node_colors = [layer_mapping[G.nodes[node]['layer']] for node in G.nodes()]
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color=node_colors, font_size=10, font_color='black',
            font_weight='bold')
    plt.title("Neural Network Structure")
    plt.axis('off')
    plt.show()

input_size = 5  # Pclass, Sex, Age, Fare, Embarked
hidden_layers = [8, 8]
output_size = 1


plot_neural_network(input_size, hidden_layers, output_size)
