from neural_network import NeuralNetwork
import plotly.graph_objects as go

def draw_neural_network(neural_network: NeuralNetwork):
    config = neural_network.layers_config
    fig = go.Figure()
    max_y = max(min(n, 10) for n in config) + 1

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_yaxes(range=[-max_y, max_y])
    fig.update_xaxes(range=[0, len(config) * 5])

    neuron_centers = []  # Store (x, y) positions for each neuron

    start_x = 0.5
    radius = 0.5

    for neuron_number in config:
        layer_centers = []
        if neuron_number <= 11:
            start_y = -(neuron_number - 1)
            for _ in range(neuron_number):
                center = (start_x, start_y)
                layer_centers.append(center)
                start_y += 2
        else:
            start_y = -10
            for _ in range(5):
                layer_centers.append((start_x, start_y))
                start_y += 2
            start_y += 2
            for _ in range(5):
                layer_centers.append((start_x, start_y))
                start_y += 2
        neuron_centers.append(layer_centers)
        start_x += 5

    for layer_idx in range(len(neuron_centers) - 1):
        for from_center in neuron_centers[layer_idx]:
            for to_center in neuron_centers[layer_idx + 1]:
                fig.add_shape(
                    type="line",
                    xref="x", yref="y",
                    x0=from_center[0], y0=from_center[1],
                    x1=to_center[0], y1=to_center[1],
                    line=dict(color="gray", width=1),
                    layer="above"
                )

    start_x = 0.5
    for neuron_number in config:
        if neuron_number <= 11:
            start_y = -(neuron_number - 1)
            for _ in range(neuron_number):
                fig.add_shape(
                    type="circle",
                    xref="x", yref="y",
                    x0=start_x - radius, y0=start_y - radius,
                    x1=start_x + radius, y1=start_y + radius,
                    fillcolor="rgba(0,200,0,1)",
                    layer="above"
                )
                start_y += 2
        else:
            start_y = -10
            for _ in range(5):
                fig.add_shape(
                    type="circle",
                    xref="x", yref="y",
                    x0=start_x - radius, y0=start_y - radius,
                    x1=start_x + radius, y1=start_y + radius,
                    fillcolor="rgba(0,200,0,1)",
                    layer="above"
                )
                start_y += 2

            start_dots_y = -0.75
            for _ in range(3):
                fig.add_shape(
                    type="circle",
                    xref="x", yref="y",
                    x0=start_x - 0.1, y0=start_dots_y - 0.1,
                    x1=start_x + 0.1, y1=start_dots_y + 0.1,
                    fillcolor="rgba(50,50,50,1)",
                    layer="above"
                )
                start_dots_y += 0.75

            start_y += 2
            for _ in range(5):
                fig.add_shape(
                    type="circle",
                    xref="x", yref="y",
                    x0=start_x - radius, y0=start_y - radius,
                    x1=start_x + radius, y1=start_y + radius,
                    fillcolor="rgba(0,200,0,1)",
                    layer="above"
                )
                start_y += 2
        start_x += 5

    fig.show()