import base64
import httpx
import math
from fasthtml.common import *
from fasthtml.svg import *
from graphviz import Digraph
import json

tailwindLink = Link(rel="stylesheet", href="assets/output.css", type="text/css")
app, rt = fast_app(
    pico=False,    
    hdrs=(tailwindLink,)
)

# -----------------------------------------------------------------------------
# rng related
# class that mimics the random interface in Python, fully deterministic,
# and in a way that we also control fully, and can also use in C, etc.

class RNG:
    def __init__(self, seed):
        self.state = seed

    def random_u32(self):
        # xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
        # doing & 0xFFFFFFFFFFFFFFFF is the same as cast to uint64 in C
        # doing & 0xFFFFFFFF is the same as cast to uint32 in C
        self.state ^= (self.state >> 12) & 0xFFFFFFFFFFFFFFFF
        self.state ^= (self.state << 25) & 0xFFFFFFFFFFFFFFFF
        self.state ^= (self.state >> 27) & 0xFFFFFFFFFFFFFFFF
        return ((self.state * 0x2545F4914F6CDD1D) >> 32) & 0xFFFFFFFF

    def random(self):
        # random float32 in [0, 1)
        return (self.random_u32() >> 8) / 16777216.0

    def uniform(self, a=0.0, b=1.0):
        # random float32 in [a, b)
        return a + (b-a) * self.random()

random = RNG(42)
# -----------------------------------------------------------------------------
# Value. Similar to PyTorch's Tensor but only of size 1 element

class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _prev=(), _op=''):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = _prev
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        out = Value(math.tanh(self.data), (self,), 'tanh')

        def _backward():
            self.grad += (1 - out.data**2) * out.grad
        out._backward = _backward

        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    def log(self):
        # (this is the natural log)
        out = Value(math.log(self.data), (self,), 'log')

        def _backward():
            self.grad += (1/self.data) * out.grad
        out._backward = _backward

        return out

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        return self * -1.0

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

# -----------------------------------------------------------------------------
# Multi-Layer Perceptron (MLP) network. Module here is similar to PyTorch's nn.Module

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin
        # color the neuron params light green (only used in graphviz visualization)
        vis_color([self.b] + self.w, "lightgreen")

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.tanh() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'TanH' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

# -----------------------------------------------------------------------------
# loss function: the negative log likelihood (NLL) loss
# NLL loss = CrossEntropy loss when the targets are one-hot vectors
# same as PyTorch's F.cross_entropy

def cross_entropy(logits, target):
    # subtract the max for numerical stability (avoids overflow)
    # commenting these two lines out to get a cleaner visualization
    # max_val = max(val.data for val in logits)
    # logits = [val - max_val for val in logits]
    # 1) evaluate elementwise e^x
    ex = [x.exp() for x in logits]
    # 2) compute the sum of the above
    denom = sum(ex)
    # 3) normalize by the sum to get probabilities
    probs = [x / denom for x in ex]
    # 4) log the probabilities at target
    logp = (probs[target]).log()
    # 5) the negative log likelihood loss (invert so we get a loss - lower is better)
    nll = -logp
    return nll


# -----------------------------------------------------------------------------
# The AdamW optimizer, same as PyTorch optim.AdamW

class AdamW:
    def __init__(self, parameters, lr=1e-1, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0):
        self.parameters = parameters
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        # state of the optimizer
        self.t = 0 # step counter
        for p in self.parameters:
            p.m = 0 # first moment
            p.v = 0 # second moment

    def step(self):
        self.t += 1
        for p in self.parameters:
            if p.grad is None:
                continue
            p.m = self.beta1 * p.m + (1 - self.beta1) * p.grad
            p.v = self.beta2 * p.v + (1 - self.beta2) * (p.grad ** 2)
            m_hat = p.m / (1 - self.beta1 ** self.t)
            v_hat = p.v / (1 - self.beta2 ** self.t)
            p.data -= self.lr * (m_hat / (v_hat ** 0.5 + 1e-8) + self.weight_decay * p.data)

    def zero_grad(self):
        for p in self.parameters:
            p.grad = 0

# -----------------------------------------------------------------------------
# data related - generates Yin Yang dataset
# Thank you https://github.com/lkriener/yin_yang_data_set

def gen_data_yinyang(random: RNG, n=1000, r_small=0.1, r_big=0.5):
    pts = []

    def dist_to_right_dot(x, y):
        return ((x - 1.5 * r_big)**2 + (y - r_big)**2) ** 0.5

    def dist_to_left_dot(x, y):
        return ((x - 0.5 * r_big)**2 + (y - r_big)**2) ** 0.5

    def which_class(x, y):
        d_right = dist_to_right_dot(x, y)
        d_left = dist_to_left_dot(x, y)
        criterion1 = d_right <= r_small
        criterion2 = d_left > r_small and d_left <= 0.5 * r_big
        criterion3 = y > r_big and d_right > 0.5 * r_big
        is_yin = criterion1 or criterion2 or criterion3
        is_circles = d_right < r_small or d_left < r_small

        if is_circles:
            return 2
        return 0 if is_yin else 1

    def get_sample(goal_class=None):
        while True:
            x = random.uniform(0, 2 * r_big)
            y = random.uniform(0, 2 * r_big)
            if ((x - r_big)**2 + (y - r_big)**2) ** 0.5 > r_big:
                continue
            c = which_class(x, y)
            if goal_class is None or c == goal_class:
                scaled_x = (x / r_big - 1) * 2
                scaled_y = (y / r_big - 1) * 2
                return [scaled_x, scaled_y, c]

    for i in range(n):
        goal_class = i % 3
        x, y, c = get_sample(goal_class)
        pts.append([[x, y], c])

    tr = pts[:int(0.8 * n)]
    val = pts[int(0.8 * n):int(0.9 * n)]
    te = pts[int(0.9 * n):]
    return tr, val, te


# -----------------------------------------------------------------------------
# visualization related functions from utils.py

def vis_color(nodes, color):
    for n in nodes:
        setattr(n, '_vis_color', color)

def trace(root):
    nodes, edges = [], []
    def build(v):
        if v not in nodes:
            nodes.append(v)
            for child in v._prev:
                if (child, v) not in edges:
                    edges.append((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root, format='svg', rankdir='LR', outfile='graph'):
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir, 'nodesep': '0.1', 'ranksep': '0.4'})

    for n in nodes:
        fillcolor = n._vis_color if hasattr(n, '_vis_color') else "white"
        dot.node(name=str(id(n)), label=f"data: {n.data:.4f}\ngrad: {n.grad:.4f}", shape='box', style='filled', fillcolor=fillcolor, width='0.1', height='0.1', fontsize='10')
        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op, width='0.1', height='0.1', fontsize='10')
            dot.edge(str(id(n)) + n._op, str(id(n)), minlen='1')

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op, minlen='1')

    dot.render(outfile, format=format)
    return f"{outfile}.{format}"


# -----------------------------------------------------------------------------
# MicrogradVisualizer

class Visualizer:
    def __init__(self):
        self.model = None
        self.train_split = None
        self.val_split = None
        self.test_split = None
        self.step_count = 0
        self.train_losses = []
        self.val_losses = []
        self.is_playing = False
        self.optimizer = None

    def reset(self):        
        self.model = None
        self.optimizer = None
        self.step_count = 0
        self.train_losses = []
        self.val_losses = []
        self.is_playing = False
        self.train_split = None
        self.val_split = None
        self.test_split = None

    def get_model_state(self):
        return {
            "parameters": [
                {
                    "data": p.data,
                    "grad": p.grad,
                    "_op": p._op if hasattr(p, '_op') else '',
                    "_prev": [id(prev) for prev in p._prev] if hasattr(p, '_prev') else []
                }
                for p in self.model.parameters()
            ],
            "structure": str(self.model)
        }

    def get_latest_losses(self):
        return (self.train_losses[-1] if self.train_losses else None,
                self.val_losses[-1] if self.val_losses else None)

    def get_training_data(self):
        return [{"x": x, "y": y} for x, y in self.train_split]

    def get_validation_data(self):
        return [{"x": x, "y": y} for x, y in self.val_split]
    
    def generate_dataset(self, n=100):
        self.reset()
        self.train_split, self.val_split, self.test_split = gen_data_yinyang(random, n=n)
        self.initialize_model()

    def initialize_model(self):
        self.model = MLP(2, [8, 3])
        self.optimizer = AdamW(self.model.parameters(), lr=1e-1, weight_decay=1e-4)

    def loss_fun(self, split):
        total_loss = Value(0.0)
        for x, y in split:
            logits = self.model(x)
            loss = cross_entropy(logits, y)
            total_loss = total_loss + loss
        mean_loss = total_loss * (1.0 / len(split))
        return mean_loss
    
    def toggle_play(self):
        self.is_playing = not self.is_playing
        return self.is_playing
    
    def train_step(self):        
        if not hasattr(self, 'train_split') or self.train_split is None or not self.model:
            self.generate_dataset()

        # forward pass
        loss = self.loss_fun(self.train_split)
        # backward pass
        loss.backward()
        # update model parameters
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.step_count += 1
        self.train_losses.append(loss.data)

        # evaluate validation loss every 10 steps
        if self.step_count % 10 == 0:
            val_loss = self.loss_fun(self.val_split)
            self.val_losses.append(val_loss.data)

        return loss.data

    def get_latest_losses(self):
        train_loss = self.train_losses[-1] if self.train_losses else None
        val_loss = self.val_losses[-1] if self.val_losses else None
        return train_loss, val_loss

    def get_train_data(self):
        return [
            {'x': point[0][0], 'y': point[0][1], 'class': point[1]}
            for point in self.train_split
        ] if self.train_split else []
    
    def get_all_data(self):
        all_data = []
        for point in self.train_split:
            all_data.append((self.point_to_dict(point), 'train'))
        for point in self.val_split:
            all_data.append((self.point_to_dict(point), 'validation'))
        for point in self.test_split:
            all_data.append((self.point_to_dict(point), 'test'))
        return all_data

    def point_to_dict(self, point):
        return {'x': point[0][0], 'y': point[0][1], 'class': point[1]}

    def get_graph_data(self):
        def get_node_id(node):
            return id(node)

        nodes = []
        edges = []
        visited = set()

        def traverse(node):
            if node in visited:
                return
            visited.add(node)

            nodes.append({
                "id": get_node_id(node),
                "value": node.data,
                "grad": node.grad,
                "label": node.label if hasattr(node, 'label') else str(node.data)
            })

            for child in node._prev:
                edges.append({
                    "from": get_node_id(child),
                    "to": get_node_id(node),
                    "op": node._op if hasattr(node, '_op') else ''
                })
                traverse(child)

        # Assuming self.model has an output node or a list of parameters
        if hasattr(self.model, 'output'):
            traverse(self.model.output)
        elif hasattr(self.model, 'parameters'):
            for param in self.model.parameters():
                traverse(param)

        return {
            "nodes": nodes,
            "edges": edges
        }

    
    def get_graph_visualization(self):
        x, y = (Value(0.0), Value(0.0)), 0
        loss = self.loss_fun([(x, y)])
        loss.backward()
        vis_color(x, "lightblue")
        
        nodes, edges = trace(loss)
        
        svg_elements = []
        for i, n in enumerate(nodes):
            fillcolor = n._vis_color if hasattr(n, '_vis_color') else "white"
            x, y = 50 + (i % 5) * 100, 50 + (i // 5) * 100
            svg_elements.append(
                G(
                    Circle(cx=x, cy=y, r=20, fill=fillcolor, stroke="black"),
                    Text(f"data: {n.data:.4f}", x=x, y=y-5, text_anchor="middle", font_size="8"),
                    Text(f"grad: {n.grad:.4f}", x=x, y=y+5, text_anchor="middle", font_size="8"),
                    id=str(id(n))
                )
            )
            if n._op:
                svg_elements.append(
                    Text(n._op, x=x, y=y-30, text_anchor="middle", font_size="10")
                )

        for n1, n2 in edges:
            x1, y1 = 50 + (nodes.index(n1) % 5) * 100, 50 + (nodes.index(n1) // 5) * 100
            x2, y2 = 50 + (nodes.index(n2) % 5) * 100, 50 + (nodes.index(n2) // 5) * 100
            svg_elements.append(
                Line(x1=x1, y1=y1, x2=x2, y2=y2, stroke="black")
            )

        return Svg(*svg_elements, width=600, height=400)


visualizer = Visualizer()

animation_script = """
let isPlaying = false;
let animationId = null;

function updateVisualization(data) {
    // Update progress tracker
    document.querySelector('#progress-tracker text:nth-child(2)').textContent = `Step ${data.step_count}/100`;
    document.querySelector('#progress-tracker text:nth-child(3)').textContent = `Train loss: ${data.train_loss.toFixed(6)}`;
    document.querySelector('#progress-tracker text:nth-child(4)').textContent = `Validation loss: ${data.val_loss ? data.val_loss.toFixed(6) : 'N/A'}`;

    // Update graph visualization
    updateGraph(data.graph_data);
}

function updateGraph(graphData) {
    // Update the SVG based on the graph data
    // This function will depend on how you want to visualize your graph
    // You might use a library like D3.js for more complex visualizations
}

async function trainStep() {
    const response = await fetch('/train_step', { method: 'POST' });
    const data = await response.json();
    updateVisualization(data);

    if (isPlaying) {
        animationId = requestAnimationFrame(trainStep);
    }
}

async function togglePlay() {
    const response = await fetch('/toggle_play', { method: 'POST' });
    const data = await response.json();
    isPlaying = data.is_playing;

    const playPauseBtn = document.getElementById('play-pause-btn');
    playPauseBtn.textContent = isPlaying ? 'Pause' : 'Play';
    playPauseBtn.classList.toggle('bg-red-500', isPlaying);
    playPauseBtn.classList.toggle('bg-green-500', !isPlaying);

    updateVisualization(data);

    if (isPlaying) {
        trainStep();
    } else {
        cancelAnimationFrame(animationId);
    }
}

document.getElementById('play-pause-btn').addEventListener('click', togglePlay);
document.getElementById('step-btn').addEventListener('click', trainStep);
"""

update_script = """
function updateGraph(graphData) {
    // Clear existing graph
    const svg = document.querySelector('#training-viz svg');
    svg.innerHTML = '';

    // Create nodes
    graphData.nodes.forEach(node => {
        const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
        g.setAttribute('id', `node-${node.id}`);

        const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        circle.setAttribute('r', '20');
        circle.setAttribute('fill', 'white');
        circle.setAttribute('stroke', 'black');

        const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
        text.textContent = node.label;
        text.setAttribute('text-anchor', 'middle');
        text.setAttribute('dominant-baseline', 'middle');

        g.appendChild(circle);
        g.appendChild(text);
        svg.appendChild(g);
    });

    // Create edges
    graphData.edges.forEach(edge => {
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute('stroke', 'black');
        line.setAttribute('marker-end', 'url(#arrowhead)');
        svg.appendChild(line);
    });

    // Position nodes and edges (you might want to use a force-directed layout algorithm for better positioning)
    // This is a simple positioning for demonstration
    const nodeElements = svg.querySelectorAll('g');
    nodeElements.forEach((el, index) => {
        const x = (index % 5) * 100 + 50;
        const y = Math.floor(index / 5) * 100 + 50;
        el.setAttribute('transform', `translate(${x}, ${y})`);
    });

    // Update edge positions
    const lineElements = svg.querySelectorAll('line');
    graphData.edges.forEach((edge, index) => {
        const fromNode = document.querySelector(`#node-${edge.from}`);
        const toNode = document.querySelector(`#node-${edge.to}`);
        const fromTransform = fromNode.getAttribute('transform');
        const toTransform = toNode.getAttribute('transform');
        const [fromX, fromY] = fromTransform.match(/\d+/g);
        const [toX, toY] = toTransform.match(/\d+/g);
        
        lineElements[index].setAttribute('x1', fromX);
        lineElements[index].setAttribute('y1', fromY);
        lineElements[index].setAttribute('x2', toX);
        lineElements[index].setAttribute('y2', toY);
    });
}
"""

@rt('/')
def get():
    return Title("Micrograd"),Div(                
        H1("Micrograd Visualizer", cls="text-3xl font-bold mb-4 w-full text-center"),
        Div(
            # Dataset Section (Left Column)
            Div(
                H2("Dataset", cls="text-xl font-bold mb-2"),
                Button("Generate Dataset", 
                       hx_post="/generate_dataset", 
                       hx_target="#dataset-content",
                       hx_swap="innerHTML",
                       cls="bg-green-500 hover:bg-green-700 text-white font-bold py-3 px-6 rounded-full mb-4 mx-auto block max-w-xs"),
                Div(
                    Div(
                        P("Click 'Generate Dataset' to begin", cls="text-gray-500 p-6"),
                        cls="aspect-square bg-white border-2 border-gray-300 rounded-lg shadow-lg flex items-center justify-center"
                    ),
                    id="dataset-content",
                    cls="overflow-x-auto"
                ),
                id="dataset-section",
                cls="w-full md:w-1/4 mb-4 md:mb-0"
            ),
            # Training and Graph Section (Right Column)
            Div(
                H2("Training and Visualization", cls="text-xl font-bold mb-2"),
                # Training Progress Section
                Div(
                    H3("Training Progress", cls="text-lg font-bold mb-2"),
                    control_buttons(),
                    Div(id="progress-tracker"),
                    Div(id="training-viz", cls="h-[200px] bg-white border-2 border-gray-300 rounded-lg shadow-lg flex items-center justify-center p-4 mb-4"),
                    id="training-section",
                    cls="mb-8"
                ),
                # Graph Visualization Section
                Div(
                    H3("Computation Graph", cls="text-lg font-bold mb-2"),
                    Div(id="graph-viz", cls="h-[200px] bg-white border-2 border-gray-300 rounded-lg shadow-lg flex items-center justify-center p-4"),
                    id="graph-section"
                ),
                id="training-and-graph-section",
                cls="w-full md:w-3/4"
            ),
            cls="flex flex-col md:flex-row gap-4 md:space-x-4 w-full max-w-6xl mx-auto"
        ),
        Script(animation_script),
        cls="flex flex-col items-center p-4 min-h-screen bg-[#e0e8d8] text-black"
    )

def control_buttons(is_playing=False):
    return Div(
        Button("Reset", id="reset-btn", hx_post="/reset", hx_target="#results", hx_swap="innerHTML",
               cls="bg-blue-500 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-full mr-2 transition duration-300 ease-in-out transform hover:scale-105", 
               title="Reset"),
         Button("Pause" if is_playing else "Play", id="play-pause-btn", hx_post="/toggle_play", 
               hx_target="#training-progress",
               hx_swap="outerHTML",
               cls=f"bg-{'red' if is_playing else 'green'}-500 hover:bg-{'red' if is_playing else 'green'}-700 text-white font-bold py-3 px-6 rounded-full mr-2 transition duration-300 ease-in-out transform hover:scale-105", 
               title="Play/Pause"),
        Button("Train Step", id="step-btn", hx_post="/train_step", hx_target="#training-viz", hx_swap="outerHTML", 
               cls="bg-blue-500 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-full transition duration-300 ease-in-out transform hover:scale-105", 
               title="Train Step"),
        cls="flex justify-center space-x-4 mb-4"
    )


def filter_buttons():
    return Div(
        Div(
            Span("Filter: ", cls="mr-2 font-bold"),
            Div(
                *[Label(
                    Input(type="radio", name="data-filter", value=value, 
                          onchange="filterDataPoints(event)",
                          checked=(value == "train"), cls="radio-input"),
                    Span(cls="radio-circle"),
                    label,
                    cls="radio-item"
                ) for value, label in [("all", "All"), ("train", "Train"), ("validation", "Validation"), ("test", "Test")]],
                cls="radio-group"
            ),
            cls="mt-1"
        ),
        # Explainer divs
        Div(
            P("The dataset is split into three parts to ensure robust model training and evaluation:"),
            P("Training set: 80 points (80%)", cls="font-semibold"),
            P("Validation set: 10 points (10%)", cls="font-semibold"),
            P("Test set: 10 points (10%)", cls="font-semibold"),
            P("This split helps us train, validate, and test our model effectively."),
            id="explainer-all",
            cls="explainer",
            style="display: none;"
        ),
        Div(
            P("The training set (80 points, 80% of the data) is used to teach the model."),
            P("The model learns to recognize patterns and make predictions based on this data."),
            P("It's crucial to have a large training set to capture the underlying patterns in the data."),
            id="explainer-train",
            cls="explainer",
            style="display: block;"
        ),
        Div(
            P("The validation set (10 points, 10% of the data) helps us tune the model."),
            P("We use this set to check the model's performance during training and adjust its parameters."),
            P("This helps prevent overfitting, where the model performs well on training data but poorly on new data."),
            id="explainer-validation",
            cls="explainer",
            style="display: none;"
        ),
        Div(
            P("The test set (10 points, 10% of the data) is used for final model evaluation."),
            P("This set is completely separate from the training process and simulates 'new' unseen data."),
            P("It gives us an unbiased estimate of how well our model will perform in real-world scenarios."),
            id="explainer-test",
            cls="explainer",
            style="display: none;"
        ),        
        id="filter-buttons",
        cls="p-4"
    )

@rt('/filter_dataset')
async def post(data_filter: str):
    all_data = visualizer.get_all_data()
    filtered_data = [d for d, set_type in all_data if data_filter == 'all' or set_type == data_filter]
    svg_content = create_svg_content(filtered_data)
    
    return (
        Svg(*svg_content, 
            width="100%", height="100%", viewBox="0 0 250 250", 
            preserveAspectRatio="xMidYMid meet",
            cls="w-full h-full"),
        Script(f"""
            document.querySelectorAll('.explainer').forEach(el => el.style.display = 'none');
            document.getElementById('explainer-{data_filter}').style.display = 'block';
        """)
    )

def create_svg_content(data):
    def create_point(point, set_type):
        x = (point['x'] + 2) * 40 + 25  # Scale and shift x coordinate
        y = 225 - (point['y'] + 2) * 40  # Scale, shift, and invert y coordinate
        color = ['#FF6B6B', '#4ECDC4', '#9B59B6'][point['class']]  # Red, Teal, Purple
        opacity = '1' if set_type == 'train' else '0.5'
        return G(
            Circle(cx=x, cy=y, r=3, fill=color, stroke="#2C3E50", stroke_width=1),
            Circle(cx=x, cy=y, r=5, fill="none", stroke=color, stroke_width=1, opacity=0.5),
            cls=f"data-point {set_type}",
            data_set=set_type,
            style=f"opacity: {opacity};"
        )

    svg_content = [
        Rect(x=0, y=0, width=250, height=250, fill="#F7F9FC"),  # Background
        *[create_point(point, set_type) for point, set_type in data],
    ]

    return svg_content

@rt('/generate_dataset')
async def post():
    visualizer.generate_dataset()
    all_data = visualizer.get_all_data()
    
    def create_point(point, set_type):
        x = (point['x'] + 2) * 40 + 25
        y = 225 - (point['y'] + 2) * 40
        color = ['#FF6B6B', '#4ECDC4', '#9B59B6'][point['class']]
        opacity = '1' if set_type == 'train' else '0.2'
        return G(
            Circle(cx=x, cy=y, r=3, fill=color, stroke="#2C3E50", stroke_width=1),
            Circle(cx=x, cy=y, r=5, fill="none", stroke=color, stroke_width=1, opacity=0.5),
            cls=f"data-point {set_type}",
            data_set=set_type,
            style=f"opacity: {opacity};"
        )

    svg_content = [
        Rect(x=0, y=0, width=250, height=250, fill="#F7F9FC"),
        *[create_point(point, set_type) for point, set_type in all_data],
    ]

    return Div(
        Div(
            Svg(*svg_content, 
                width="100%", height="100%", viewBox="0 0 250 250", 
                preserveAspectRatio="xMidYMid meet",
                cls="w-full h-full"),
            cls="aspect-square bg-white border-2 border-gray-300 rounded-lg shadow-lg flex items-center justify-center mb-4"
        ),
        filter_buttons(),
        Script("""
        function filterDataPoints(event) {
            const filter = event.target.value;
            const dataPoints = document.querySelectorAll('.data-point');
            dataPoints.forEach(point => {
                point.style.opacity = (filter === 'all' || point.dataset.set === filter) ? '1' : '0.2';
            });
            
            document.querySelectorAll('.explainer').forEach(el => el.style.display = 'none');
            document.getElementById(`explainer-${filter}`).style.display = 'block';
        }
        // Trigger initial filter
        document.querySelector('input[name="data-filter"]:checked').dispatchEvent(new Event('change'));
        """),
        id="dataset-content",
        cls="w-full"
    )

# @rt('/train_step')
# async def post():
#     continued = visualizer.train_step()
#     if not continued:
#         return "Training complete"    
#     graph_svg = visualizer.get_graph_visualization()

#     train_loss, val_loss = visualizer.get_latest_losses()
    
#     return Div(        
#         graph_svg,
#         id="training-viz",
#         cls="w-full h-[500px] bg-white border-2 border-gray-300 rounded-lg shadow-lg overflow-hidden p-8"
#     ), Div(
#         progress_tracker(
#             visualizer.step_count,
#             f"{train_loss:.6f}" if train_loss is not None else "---",
#             f"{val_loss:.6f}" if val_loss is not None else "N/A"
#         ),
#         id="progress-tracker",
#         hx_swap_oob="true"
#     )


@rt('/train_step')
async def post():
    visualizer.train_step()
    return {
        "step_count": visualizer.step_count,
        "train_loss": visualizer.get_latest_losses()[0],
        "val_loss": visualizer.get_latest_losses()[1],
        "graph_data": visualizer.get_graph_data()
    }


def progress_tracker(step_count, train_loss, val_loss):
    return Div(
        Svg(
            Rect(x=0, y=0, width=600, height=50, fill="#f0f0f0", stroke="#000000"),
            Text(f"Step {step_count}/100", x=10, y=30, font_size="16", font_weight="bold"),
            Text(f"Train loss: {train_loss}", x=150, y=30, font_size="14"),
            Text(f"Validation loss: {val_loss}", x=350, y=30, font_size="14"),
            width="100%", height="50",
            preserveAspectRatio="xMidYMid meet",
            viewBox="0 0 600 50",
        ),        
        cls="w-full bg-gray-100 rounded-t-lg shadow-md",
        id="progress-tracker"
    )

@rt('/reset')
async def post():
    visualizer.reset()
    visualizer.initialize_model()
    return Div(
        Div(
            progress_tracker(0, "---", "---"),
            id="progress-tracker",            
        ),
        Div(
            P("Training progress reset. Click 'Train Step' or 'Play' to start training.", 
              cls="text-gray-600"),
            id="training-viz",
            cls="w-full h-[450px] bg-white border-2 border-gray-300 rounded-lg shadow-lg flex items-center justify-center p-8"
        ),
        id="results"
    )


@rt('/toggle_play')
async def post():
    is_playing = visualizer.toggle_play()
    return {
        "is_playing": is_playing,
        "step_count": visualizer.step_count,
        "train_loss": visualizer.get_latest_losses()[0],
        "val_loss": visualizer.get_latest_losses()[1],
        "graph_data": visualizer.get_graph_data()
    }


# @rt('/toggle_play')
# async def post():
#     is_playing = visualizer.toggle_play()
#     button_text = "Pause" if is_playing else "Play"
#     button_color = "red" if is_playing else "green"
    
#     train_loss, val_loss = visualizer.get_latest_losses()
    
#     return Div(
#         control_buttons(is_playing),
#         Div(
#             Div(progress_tracker(
#                 visualizer.step_count,
#                 f"{train_loss:.6f}" if train_loss is not None else "---",
#                 f"{val_loss:.6f}" if val_loss is not None else "N/A"
#             ), id="progress-tracker"),
#             Div(
#                 visualizer.get_graph_visualization(),
#                 id="training-viz",
#                 cls="w-full h-[450px] bg-white border-2 border-t-0 border-gray-300 rounded-b-lg shadow-lg overflow-hidden p-8"
#             ),
#             id="results",
#             cls="w-full"
#         ),
#         id="training-progress",
#         hx_target="#training-progress",
#         hx_swap="outerHTML",
#         cls="w-full"
#     )

# Run the app
serve()