import math
from fasthtml.common import *
from fasthtml.svg import *
from graphviz import Digraph
from pydantic import BaseModel
from typing import List, Dict, Any
import json
import asyncio
import time

tailwindLink = Link(rel="stylesheet", href="assets/output.css", type="text/css")
sselink = Script(src="https://unpkg.com/htmx-ext-sse@2.2.1/sse.js")
chartlink = Script(src="https://cdn.jsdelivr.net/npm/chart.js")
svgpanzoomlink = Script(src="https://cdn.jsdelivr.net/npm/svg-pan-zoom@3.6.1/dist/svg-pan-zoom.min.js")
app, rt = fast_app(
    pico=False,    
    hdrs=(tailwindLink, sselink, chartlink, svgpanzoomlink)
)

setup_toasts(app)


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
# The AdamW optimizer, same as PyTorch optim.AdamW (https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)

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
        self.step_count = 0
        self.train_losses = []
        self.val_losses = []
        self.train_split = []
        self.val_split = []
        self.test_split = []
        self.batch_size = 20
        self.is_training = False
        self.history = []
        self.initialize_model()

    def initialize_model(self):
        self.model = MLP(2, [8, 3])
        self.optimizer = AdamW(self.model.parameters(), lr=1e-1, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)

    def get_optimizer_state(self):
        return {
            'lr': self.optimizer.lr,
            'beta1': self.optimizer.beta1,
            'beta2': self.optimizer.beta2,
            'eps': self.optimizer.eps,
            'weight_decay': self.optimizer.weight_decay,            
            't': self.optimizer.t
        }
    
    def reset(self):
        if self.model is None:
            self.initialize_model()
        
        # Reset training progress
        self.is_training = False
        self.step_count = 0
        self.train_losses = []
        self.val_losses = []
        self.history = []
        
        # Reset model parameters to initial values
        for p in self.model.parameters():
            p.data = random.uniform(-0.1, 0.1)
        
        # Reset optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=1e-1, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)

        # Reset progress tracker
        self.progress_tracker = {
            'step_count': 0,
            'train_loss': "---",
            'val_loss': "---"
        }

    def generate_dataset(self, n=100):        
        self.train_split, self.val_split, self.test_split = gen_data_yinyang(random, n=n)
        # Initialize the model if it doesn't exist
        if self.model is None:
            self.initialize_model()
        
        # Now reset the training progress
        self.reset()

    def train_step(self):
        if not self.model or not self.train_split or not self.is_training:
            return self.get_current_step()
        
        if self.is_training:
            # Perform one training step
            X_train, y_train = zip(*self.train_split)
            loss = self.loss_fun(X_train, y_train)
            loss.backward()
            self.optimizer.step()
            current_step = self.get_current_step()
            self.history.append(current_step)

            self.optimizer.zero_grad()

            self.step_count += 1
            self.train_losses.append(loss.data)

            # Evaluate validation loss every 10 steps
            if self.step_count % 10 == 0 and self.val_split:
                X_val, y_val = zip(*self.val_split)
                val_loss = self.loss_fun(X_val, y_val)
                self.val_losses.append(val_loss.data)
                                            
        return current_step

    def get_current_step(self):
        param_state = {}
        if self.model:
            for i, p in enumerate(self.model.parameters()):
                param_state[f'param_{i}'] = {
                    'value': p.data.item() if hasattr(p.data, 'item') else float(p.data),
                    'grad': p.grad.item() if (p.grad is not None and hasattr(p.grad, 'item')) else float(p.grad) if p.grad is not None else 0,
                    'm': float(getattr(p, 'm', 0)),
                    'v': float(getattr(p, 'v', 0))
                }
        return {
            'step_count': self.step_count,
            'train_loss': f"{self.train_losses[-1]:.6f}" if self.train_losses else "---",
            'val_loss': f"{self.val_losses[-1]:.6f}" if self.val_losses and len(self.val_losses) > 0 else "---",
            'is_training': self.is_training,
            'param_state': param_state
        }

    def loss_fun(self, X, y):
        total_loss = Value(0.0)
        for x, y in zip(X, y):
            logits = self.model(x)
            loss = cross_entropy(logits, y)
            total_loss = total_loss + loss
        mean_loss = total_loss * (1.0 / len(X))
        return mean_loss
    
    def update_param_state(self, param_state):
        if self.model:
            for i, p in enumerate(self.model.parameters()):
                param_name = f'param_{i}'
                if param_name in param_state:
                    p.data = param_state[param_name]['value']
                    p.grad = param_state[param_name]['grad']
                    setattr(p, 'm', param_state[param_name]['m'])
                    setattr(p, 'v', param_state[param_name]['v'])

    def get_graph_data(self):
        nodes = []
        edges = []
        visited = set()

        def traverse(node):
            if node in visited:
                return
            visited.add(node)

            nodes.append({
                "id": id(node),
                "value": node.data,
                "grad": node.grad
            })

            for child in node._prev:
                edges.append({
                    "from": id(child),
                    "to": id(node),
                    "grad": node.grad if hasattr(node, '_backward') else None
                })
                traverse(child)

        if hasattr(self.model, 'output'):
            traverse(self.model.output)
        elif hasattr(self.model, 'parameters'):
            for param in self.model.parameters():
                traverse(param)

        return {
            "nodes": nodes,
            "edges": edges
        }

    def get_all_data(self):
        all_data = []
        for point in self.train_split:
            all_data.append((self.point_to_dict(point), 'train'))
        for point in self.val_split:
            all_data.append((self.point_to_dict(point), 'validation'))
        return all_data

    def point_to_dict(self, point):
        return {'x': point[0][0], 'y': point[0][1], 'class': point[1]}

visualizer = Visualizer()

@rt('/train_stream')
async def get(request):
    async def event_stream():
        heartbeat_interval = 30  # Send heartbeat every 30 seconds when not training
        last_heartbeat = 0
        while True:
            train = request.query_params.get('train', '').lower() == 'true'
            step = request.query_params.get('step', '').lower() == 'true'
            
            if train or step:
                visualizer.is_training = True
                step_data = visualizer.train_step()
                visualizer.is_training = False if step else True
                yield f"event: step\ndata: {json.dumps(step_data)}\n\n"
                if step:
                    visualizer.is_training = False
                    break  # Exit the loop after a single step
                await asyncio.sleep(0.3)  # 300ms delay between training steps
            else:
                visualizer.is_training = False
                current_time = time.time()
                if current_time - last_heartbeat >= heartbeat_interval:
                    yield f"event: heartbeat\ndata: {{}}\n\n"
                    last_heartbeat = current_time
                await asyncio.sleep(1)  # Check every second, but only send heartbeat at interval

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@rt('/progress_tracker')
async def get():
    return progress_tracker(visualizer.step_count,
                            f"{visualizer.train_losses[-1]:.6f}" if visualizer.train_losses else "---",
                            f"{visualizer.val_losses[-1]:.6f}" if visualizer.val_losses else "---")

def progress_tracker(step_count, train_loss, val_loss):
    return Svg(
        Rect(x=0, y=0, width=600, height=50, fill="#f0f0f0", stroke="#000000"),
        Text(f"Step {step_count}/100", x=10, y=30, font_size="16", font_weight="bold", id="step-text"),
        Text(f"Train loss: {train_loss:.6f}" if isinstance(train_loss, float) else f"Train loss: {train_loss}", x=150, y=30, font_size="14", id="train-loss-text", data_value=train_loss),
        Text(f"Validation loss: {val_loss:.6f}" if isinstance(val_loss, float) else f"Validation loss: {val_loss}", x=350, y=30, font_size="14", id="val-loss-text", data_value=val_loss),
        width="100%", height="50",
        preserveAspectRatio="xMidYMid meet",
        viewBox="0 0 600 50",
        id="progress-tracker",
        hx_trigger="progressUpdate"
    )

@rt("/update-optimizer-state")
async def update_optimizer_state(request: Request):
    data = await request.json()
    # Assuming you have a global or accessible visualizer object
    visualizer.update_param_state(data)
    return create_main_parameter_dials(visualizer)

animation_script = """
const config = {
    maxReconnectAttempts: 5,
    reconnectDelay: 5000,
    maxVisibleSteps: 25
};

let state = {
    isTraining: false,
    eventSource: null,
    reconnectAttempts: 0,    
};

const elements = {
    playPauseBtn: () => document.getElementById('play-pause-btn'),
    stepText: () => document.getElementById('step-text'),
    trainLossText: () => document.getElementById('train-loss-text'),
    valLossText: () => document.getElementById('val-loss-text'),
    chartContainer: () => document.getElementById('loss-chart-container'),
    parameterDials: () => document.getElementById('parameter-dials')
};

const updateUI = {
    playPauseButton: () => {
        const btn = elements.playPauseBtn();
        btn.textContent = state.isTraining ? 'Pause' : 'Play';
        btn.classList.toggle('bg-red-500', state.isTraining);
        btn.classList.toggle('bg-green-500', !state.isTraining);
    },
    progressTracker: (data) => {
        if (data.step_count !== undefined) {
            elements.stepText().textContent = `Step ${data.step_count}/100`;
        }
        if (data.train_loss !== undefined) {
            elements.trainLossText().textContent = `Train loss: ${data.train_loss}`;
            elements.trainLossText().dataset.value = data.train_loss;
        }
        if (data.val_loss !== undefined) {
            elements.valLossText().textContent = `Validation loss: ${data.val_loss}`;
            elements.valLossText().dataset.value = data.val_loss;
        }
        elements.chartContainer().classList.toggle('hidden', data.step_count === 0);
        if (data.step_count > 0) {
            updateLossChart(data.step_count, parseFloat(data.train_loss), parseFloat(data.val_loss));
        }
        if (data.param_state !== undefined) {
            updateOptimizerState(data.param_state);
        }
        document.dispatchEvent(new CustomEvent('progressUpdated', { detail: data }));
    }
};

const updateSSEConnection = () => {
    if (state.eventSource) state.eventSource.close();
    
    if (state.reconnectAttempts >= config.maxReconnectAttempts) {
        console.error('Max reconnection attempts reached. Please refresh the page.');
        return;
    }

    state.eventSource = new EventSource(`/train_stream?train=${state.isTraining}`);
    
    state.eventSource.addEventListener('open', () => {
        console.log('SSE connection opened');
        state.reconnectAttempts = 0;
    });

    state.eventSource.addEventListener('step', (e) => {
        updateUI.progressTracker(JSON.parse(e.data));
    });

    state.eventSource.addEventListener('heartbeat', () => console.log('Heartbeat received'));

    state.eventSource.addEventListener('error', (e) => {
        console.error('SSE connection error:', e);
        state.eventSource.close();
        state.reconnectAttempts++;
        setTimeout(updateSSEConnection, config.reconnectDelay);
    });
};

const handlePlayPause = () => {
    state.isTraining = !state.isTraining;
    updateUI.playPauseButton();
    updateSSEConnection();
};

const handleTrainStep = () => {
    if (!state.isTraining) {
        fetch('/train_stream?step=true')
            .then(response => response.text())
            .then(text => {
                const eventData = text.split('\\n').find(line => line.startsWith('data:'));
                if (eventData) {
                    updateUI.progressTracker(JSON.parse(eventData.slice(5)));
                }
            })
            .catch(error => console.error('Error during single step:', error));
    }
};

const handleReset = () => {
    fetch('/reset', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            updateUI.progressTracker(data);
            state.isTraining = false;
            updateUI.playPauseButton();
            updateSSEConnection();
            console.log('Reset to step 0');
            resetChart();
        });
};

const updateOptimizerState = (paramState) => {
    fetch('/update-optimizer-state', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(paramState)
    })
    .then(response => response.text())
    .then(svg => {
        const parameterDials = elements.parameterDials();
        if (parameterDials) {
            parameterDials.innerHTML = svg;
            // Dispatch custom event after updating SVG
            document.body.dispatchEvent(new CustomEvent('svgUpdated'));
        } else {
            console.error("Element with id 'parameter-dials' not found");
        }
    })
    .catch(error => console.error('Error updating optimizer state:', error));
};

let lossChart;

function initChart() {
    const ctx = document.getElementById('loss-chart');
    if (ctx && !lossChart) {
        lossChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Training Loss',
                    data: [],
                    backgroundColor: 'rgba(75, 192, 192, 0.6)',
                    borderColor: 'rgb(75, 192, 192)',
                    borderWidth: 1
                },
                {
                    label: 'Validation Loss',
                    data: [],
                    backgroundColor: 'rgba(255, 99, 132, 0.6)',
                    borderColor: 'rgb(255, 99, 132)',
                    borderWidth: 1
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        stacked: true,
                        beginAtZero: true,
                        position: 'top',
                        title: {
                            display: true,
                            text: 'Loss',
                            padding: {top: 10, bottom: 0}
                        }
                    },
                    y: {
                        stacked: true,
                        reverse: true,
                        title: {
                            display: true,
                            text: 'Steps',
                            padding: {top: 0, left: 10, right: 10, bottom: 0}
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    annotation: {
                        annotations: {
                            currentStep: {
                                type: 'line',
                                yMin: 0,
                                yMax: 0,
                                borderColor: 'rgb(255, 255, 0)',
                                borderWidth: 2
                            }
                        }
                    }
                }
            }
        });
    }
}

function resetChart() {
if (lossChart) {
lossChart.data.labels = [];
lossChart.data.datasets[0].data = [];
lossChart.data.datasets[1].data = [];
lossChart.update();
}

const chartContainer = document.getElementById('loss-chart-container');
if (chartContainer) {
chartContainer.classList.add('hidden');
}    
}

function updateLossChart(stepCount, trainLoss, valLoss) {
    const chartContainer = document.getElementById('loss-chart-container');
    if (stepCount > 0) {
        chartContainer.classList.remove('hidden');
        if (lossChart) {
            const maxVisibleSteps = 25;
            
            lossChart.data.labels.push(stepCount);
            lossChart.data.datasets[0].data.push(trainLoss);
            lossChart.data.datasets[1].data.push(valLoss);
            
            if (lossChart.data.labels.length > maxVisibleSteps) {
                lossChart.data.labels = lossChart.data.labels.slice(-maxVisibleSteps);
                lossChart.data.datasets[0].data = lossChart.data.datasets[0].data.slice(-maxVisibleSteps);
                lossChart.data.datasets[1].data = lossChart.data.datasets[1].data.slice(-maxVisibleSteps);
            }
            
            lossChart.options.plugins.annotation.annotations.currentStep.yMin = lossChart.data.labels.length - 1;
            lossChart.options.plugins.annotation.annotations.currentStep.yMax = lossChart.data.labels.length - 1;
            
            lossChart.update();
        }                                
    } else {
        chartContainer.classList.add('hidden');
    }
}

document.body.addEventListener('htmx:sseMessage', (evt) => {
    if (evt.detail.type === 'step') {
        updateUI.progressTracker(JSON.parse(evt.detail.data));
    }
});

['play-pause-btn', 'step-btn', 'reset-btn'].forEach(id => {
    document.getElementById(id).addEventListener('click', {
        'play-pause-btn': handlePlayPause,
        'step-btn': handleTrainStep,
        'reset-btn': handleReset
    }[id]);
});

window.addEventListener('beforeunload', () => {
    if (state.eventSource) state.eventSource.close();
});

document.addEventListener('DOMContentLoaded', () => {
    updateSSEConnection();
    initChart();
    initializeSvgPanZoom();
});
"""

# -----------------------------------------------------------------------------
# svg-pan-zoom related
panzoomscript = """
let panZoomInstance = null;

function initializeSvgPanZoom() {
    const svgPanZoomState = JSON.parse(document.getElementById('panZoomState').value);
    const container = document.getElementById('parameter-dials');
    const svgElement = container.querySelector('svg');
    if (svgElement) {
        // Set the SVG background to transparent
        svgElement.style.backgroundColor = 'transparent';
        
        if (panZoomInstance) {
            panZoomInstance.destroy();
        }
        
        panZoomInstance = svgPanZoom(svgElement, {
            zoomEnabled: true,
            controlIconsEnabled: true,
            fit: true,            
            center: true,
            minZoom: 0.1,
            maxZoom: 10
        });
        panZoomInstance.setOnPan(function(point) {        
        document.getElementById('panZoomState').value = JSON.stringify({ zoom: panZoomInstance.getZoom(), pan: point });
        });
        panZoomInstance.setOnZoom(function(zoom) {
            document.getElementById('panZoomState').value = JSON.stringify({ zoom: zoom, pan: panZoomInstance.getPan() });
        });

        
        panZoomInstance.zoom(svgPanZoomState.zoom);
        panZoomInstance.pan(svgPanZoomState.pan);
    

        // Resize and fit on window resize
        window.addEventListener('resize', function() {
            panZoomInstance.resize();
            panZoomInstance.fit();
            panZoomInstance.center();
        });
    }
}

// Listen for the custom event that will be triggered after the SVG is updated
document.body.addEventListener('svgUpdated', function() {
    initializeSvgPanZoom();
});
"""
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def create_hyperparameter_controls(optimizer_state):
    return Div(
        Div(
            Label("lr:", For="lr-select", cls="mr-2"),
            Select(
                Option(f"{optimizer_state['lr']:.4f}", value=optimizer_state['lr'], selected=True),
                id="lr-select", disabled=True, cls="bg-gray-100 border border-gray-300 rounded-md p-1"
            ),
            cls="mb-2"
        ),
        Div(
            Label("Beta 1:", For="beta1-select", cls="mr-2"),
            Select(
                Option(f"{optimizer_state['beta1']:.4f}", value=optimizer_state['beta1'], selected=True),
                id="beta1-select", disabled=True, cls="bg-gray-100 border border-gray-300 rounded-md p-1"
            ),
            cls="mb-2"
        ),
        Div(
            Label("Beta 2:", For="beta2-select", cls="mr-2"),
            Select(
                Option(f"{optimizer_state['beta2']:.4f}", value=optimizer_state['beta2'], selected=True),
                id="beta2-select", disabled=True, cls="bg-gray-100 border border-gray-300 rounded-md p-1"
            ),
            cls="mb-2"
        ),
        Div(
            Label("eps:", For="eps-select", cls="mr-2"),
            Select(
                Option(f"{optimizer_state['eps']:.4e}", value=optimizer_state['eps'], selected=True),
                id="eps-select", disabled=True, cls="bg-gray-100 border border-gray-300 rounded-md p-1"
            ),
            cls="mb-2"
        ),
        Div(
            Label("Weight Decay:", For="weight-decay-select", cls="mr-2"),
            Select(
                Option(f"{optimizer_state['weight_decay']:.4e}", value=optimizer_state['weight_decay'], selected=True),
                id="weight-decay-select", disabled=True, cls="bg-gray-100 border border-gray-300 rounded-md p-1"
            ),
            cls="mb-2"
        ),
        cls="flex flex-wrap gap-4 mb-4"
    )

def create_main_parameter_dials(visualizer, container_width=800, container_height=400):
    if not visualizer.model:
        return Div("No model available", id="parameter-dials")

    dial_size = 60
    dial_gap = 10
    
    total_params = sum(len(neuron.w) + 1 for layer in visualizer.model.layers for neuron in layer.neurons)
    columns = max(1, (container_width + dial_gap) // (dial_size + dial_gap))
    rows = math.ceil(total_params / columns)
    
    grid_width = columns * (dial_size + dial_gap) - dial_gap
    grid_height = rows * (dial_size + dial_gap) - dial_gap
    
    start_x = (container_width - grid_width) / 2
    start_y = (container_height - grid_height) / 2    
    
    optimizer_state = visualizer.get_optimizer_state()
    current_step = visualizer.get_current_step()

    dials = []
    param_index = 0
    for i, layer in enumerate(visualizer.model.layers):
        for j, neuron in enumerate(layer.neurons):
            for k, w in enumerate(neuron.w):
                param_state = current_step['param_state'][f'param_{param_index}']
                dials.append(resize_dial(f"layer{i}_neuron{j}_weight{k}", w, dial_size, 
                                         param_state['grad'], param_state['m'], param_state['v'],
                                         optimizer_state))
                param_index += 1
            param_state = current_step['param_state'][f'param_{param_index}']
            dials.append(resize_dial(f"layer{i}_neuron{j}_bias", neuron.b, dial_size,
                                     param_state['grad'], param_state['m'], param_state['v'],
                                     optimizer_state))
            param_index += 1
    
    positioned_dials = [
        G(dial, transform=f"translate({start_x + (i % columns) * (dial_size + dial_gap)}, {start_y + (i // columns) * (dial_size + dial_gap)})")
        for i, dial in enumerate(dials[:columns * rows])
    ]

    svg_width = max(container_width, grid_width + 2 * start_x)
    svg_height = max(container_height, grid_height + 2 * start_y)
    
    return Svg(
        Rect(x=0, y=0, width=svg_width, height=svg_height, fill="#f0f0f0"),
        *positioned_dials,        
        width="100%", height="100%", 
        viewBox=f"0 0 {svg_width} {svg_height}",
        preserveAspectRatio="xMidYMid meet",
        id="parameter-dials-svg",
        style="min-height: 400px;"
    )

def resize_dial(param_name, param, size, gradient, m, v, optimizer_state):
    dial = create_parameter_dial(
        param_name=param_name,
        data=float(param.data),
        gradient=float(gradient),
        m=float(m),
        v=float(v),
        optimizer_state=optimizer_state
    )
    if dial is not None:
        dial.attrs['width'] = size
        dial.attrs['height'] = size
        return dial
    else:
        print(f"Warning: Dial for {param_name} is None")
        return None

@rt('/')
def get():
    return Title("Micrograd Visualizer"),Div(                
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
                    Div(
                    Canvas(id="loss-chart"),
                    id="loss-chart-container",
                    cls="w-full aspect-square bg-white border-2 border-gray-300 rounded-lg shadow-lg hidden"
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
                    control_buttons(visualizer.is_training),                    
                    Div(
                        progress_tracker(0, "---", "---"),       
                        id="training-content",
                        hx_ext="sse",
                        sse_connect="/train_stream",
                        sse_swap="step",
                        cls="bg-white border-2 border-gray-300 rounded-lg shadow-lg p-4 mb-4 w-full flex flex-col items-stretch"),                    
                    id="training-section",
                    cls="mb-8"
                ),
                # Graph Visualization Section
                Div(
                    H3("Optimizer State", cls="text-xl font-semibold mb-2"),
                    create_hyperparameter_controls(visualizer.get_optimizer_state()),
                     Div(
                        create_main_parameter_dials(visualizer),
                        id="parameter-dials",
                        cls="w-full h-[400px] overflow-hidden"
                    ),
                    Input(type="hidden", id="panZoomState", value='{"zoom": 1, "pan": {"x": 0, "y": 0}}'),
                    id="optimizer-state-container",
                    cls="bg-white p-4 rounded-lg shadow-md"
                ),
                id="training-and-graph-section",
                cls="w-full md:w-3/4"
            ),
            cls="flex flex-col md:flex-row gap-4 md:space-x-4 w-full max-w-6xl mx-auto"
        ),         
        Script(animation_script),
        Script(panzoomscript),
        cls="flex flex-col items-center p-4 min-h-screen bg-[#e0e8d8] text-black"
    )

def control_buttons(is_training=False):
    return Div(
        Button("Reset", id="reset-btn", 
               cls="bg-blue-500 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-full mr-2 transition duration-300 ease-in-out transform hover:scale-105", 
               title="Reset"),
        Button("Pause" if is_training else "Play", id="play-pause-btn",
               cls=f"{'bg-red-500' if is_training else 'bg-green-500'} hover:bg-green-700 text-white font-bold py-3 px-6 rounded-full mr-2 transition duration-300 ease-in-out transform hover:scale-105", 
               title="Play/Pause"),
        Button("Train Step", id="step-btn",
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
        Details(
            Summary("Dataset Info", cls="text-lg font-bold cursor-pointer"),
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
        ),    
        id="filter-buttons",
        cls="p-4"
    )

@rt('/filter_dataset')
async def post(data_filter: str):
    all_data = visualizer.get_all_data()
    filtered_data = [d for d, set_type in all_data if data_filter == 'all' or set_type == data_filter]
    svg_content = create_dataset_svg(filtered_data)
    
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

def create_dataset_svg(data):
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
            resetChart();

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

@rt('/reset')
async def post():
    visualizer.reset()
    return {
        'step_count': 0,
        'train_loss': "---",
        'val_loss': "---"
    }

# ------------------------------------------------------------------------------------------
# -------------------------AdamW Optimizer----------------------------------------------
# ------------------------------------------------------------------------------------------

@rt('/adamw')
async def get():    
    return Div(
        Link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css", integrity="sha384-nB0miv6/jRmo5UMMR1wu3Gz6NLsoTkbqJghGIsx//Rlm+ZU03BU6SQNC66uf4l5+", crossorigin="anonymous"),
        Script(src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js", integrity="sha384-7zkQWkzuo3B5mTepMUcHkMB5jZaolc2xDwL6VFqjFALcbeS9Ggm/Yr2r3Dy4lfFg", crossorigin="anonymous"),
        Script("""
            document.addEventListener("DOMContentLoaded", function() {
                document.querySelectorAll('.math').forEach(function(element) {
                    katex.render(element.textContent, element, {
                        throwOnError: false
                    });
                });
            });
        """),              
        Div(
            H1("AdamW Optimizer", cls="text-3xl font-bold mb-6 text-center text-blue-600"),
            
            # Overall View
            Div(
           H2("AdamW in Action", cls="text-2xl font-semibold mb-4"),
            P("Watch how AdamW optimizes parameters over multiple steps:", cls="mb-4"),
            Div(                                
                id="adamw-visualization-container", 
                cls="mb-4"
            ),
            Button("Run Optimization", 
                id="run-optimization-btn",
                hx_post="/run_adamw", 
                hx_target="#visualization-container",
                hx_trigger="click",
                hx_sse="true",
                cls="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded text-sm"),
            cls="bg-gray-100 p-6 rounded-lg shadow-md mb-8"
        ),
            
            # Step-by-Step Breakdown
            Div(
                H2("Algorithm Breakdown", cls="text-2xl font-semibold mb-4"),
                Div(id="step-breakdown", cls="mb-4"),
                Button("Next Step", hx_post="/next_step", hx_target="#step-breakdown", cls="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded text-sm"),
                cls="bg-gray-100 p-6 rounded-lg shadow-md mb-8"
            ),
            
            # Hyperparameter Deep Dives
            Div(
                H2("Explore Hyperparameters", cls="text-2xl font-semibold mb-4"),
                Div(
                    Button("Learning Rate", hx_get="/hyperparam/lr", hx_target="#hyperparam-details"),
                    Button("Beta Values", hx_get="/hyperparam/betas", hx_target="#hyperparam-details"),
                    Button("Epsilon", hx_get="/hyperparam/epsilon", hx_target="#hyperparam-details"),
                    Button("Weight Decay", hx_get="/hyperparam/weight_decay", hx_target="#hyperparam-details"),
                    cls="flex flex-wrap gap-2 mb-4"
                ),
                Div(id="hyperparam-details", cls="mt-4"),
                cls="bg-gray-100 p-6 rounded-lg shadow-md mb-8"
            ),
            
            # Advanced Interactions (initially hidden, can be revealed later)
            Div(id="advanced-interactions", cls="hidden"),
            
            cls="max-w-6xl mx-auto px-4 py-8"
        ),
        cls="mx-auto min-h-screen bg-[#e0e8d8] text-gray-800"
    )




    #         Div(
    #             H2("Inputs and Initialization", cls="text-2xl font-semibold mb-4 text-blue-500"),
    #             P("AdamW is initialized with the following parameters:", cls="mb-4"),
    #             Ul(
    #                 Li(Span("lr (\\alpha)", cls="math font-semibold"), ": learning rate", cls="mb-2"),
    #                 Li(Span("\\beta_1, \\beta_2", cls="math font-semibold"), ": coefficients for computing running averages of gradient and its square", cls="mb-2"),
    #                 Li(Span("eps", cls="math font-semibold"), ": term added to the denominator to improve numerical stability", cls="mb-2"),
    #                 Li(Span("weight\\_decay", cls="math font-semibold"), ": weight decay (L2 penalty)", cls="mb-2"),
    #                 cls="list-disc list-inside mb-6"
    #             ),
    #             P("Initialize:", cls="mb-4"),
    #             Ul(
    #                 Li(Span("m_0 = 0", cls="math"), " (first moment vector)", cls="mb-2"),
    #                 Li(Span("v_0 = 0", cls="math"), " (second moment vector)", cls="mb-2"),
    #                 Li(Span("\\hat{v}_0^{max} = 0", cls="math"), " (if using AMSGrad)", cls="mb-2"),
    #                 cls="list-disc list-inside mb-6"
    #             ),
    #             cls="bg-gray-100 p-6 rounded-lg shadow-md mb-8"
    #         ),
    #         Div(
    #             H2("Update Step", cls="text-2xl font-semibold mb-4 text-blue-500"),
    #             P("For each parameter ", Span("\\theta", cls="math"), " and its gradient ", Span("g", cls="math"), ":", cls="mb-4"),
    #             Ol(
    #                 Li("Apply weight decay:", 
    #                    Div(Span("\\theta_t \\leftarrow \\theta_{t-1} - lr \\cdot weight\\_decay \\cdot \\theta_{t-1}", cls="math block my-2 text-lg")), 
    #                    cls="mb-4"),
    #                 Li("Update biased first moment estimate:", 
    #                    Div(Span("m_t = \\beta_1 \\cdot m_{t-1} + (1 - \\beta_1) \\cdot g_t", cls="math block my-2 text-lg")), 
    #                    cls="mb-4"),
    #                 Li("Update biased second raw moment estimate:", 
    #                    Div(Span("v_t = \\beta_2 \\cdot v_{t-1} + (1 - \\beta_2) \\cdot g_t^2", cls="math block my-2 text-lg")), 
    #                    cls="mb-4"),
    #                 Li("Correct bias in first moment:", 
    #                    Div(Span("\\hat{m}_t = \\frac{m_t}{1 - \\beta_1^t}", cls="math block my-2 text-lg")), 
    #                    cls="mb-4"),
    #                 Li("Correct bias in second moment:", 
    #                    Div(Span("\\hat{v}_t = \\frac{v_t}{1 - \\beta_2^t}", cls="math block my-2 text-lg")), 
    #                    cls="mb-4"),
    #                 Li("If using AMSGrad, update max of ̂v:", 
    #                    Div(Span("\\hat{v}_t^{max} = max(\\hat{v}_t^{max}, \\hat{v}_t)", cls="math block my-2 text-lg")), 
    #                    cls="mb-4"),
    #                 Li("Compute the parameter update:", 
    #                    Div(Span("\\theta_t \\leftarrow \\theta_t - lr \\cdot \\frac{\\hat{m}_t}{\\sqrt{\\hat{v}_t^{max}} + eps}", cls="math block my-2 text-lg")), 
    #                    cls="mb-4"),
    #                 cls="list-decimal list-inside mb-6"
    #             ),
    #             cls="bg-gray-100 p-6 rounded-lg shadow-md"
    #         ),
    #         cls="max-w-4xl mx-auto px-4 py-8"
    #     ),
    #     cls="mx-auto min-h-screen bg-[#e0e8d8] text-gray-800"
    # )


# ------------------------------------------------------------------------------------------
# ----------------------------------------Dial--------------------------------------------
# ------------------------------------------------------------------------------------------

def create_parameter_dial(param_name, data=1.0000, gradient=25.0000, m=0.1, v=1.0, optimizer_state=None):
    SIZE = 200
    RING_WIDTH = SIZE / 8
    OUTER_RADIUS = SIZE / 2 - 5
    INNER_RADIUS = OUTER_RADIUS - RING_WIDTH
    CENTER = SIZE / 2
    SCALE_MIN, SCALE_MAX = -1.0, 1.0
    LINTHRESH = 0.01
         
    t = optimizer_state['t']    
    if t > 0:
        m_hat = m / (1 - optimizer_state['beta1'] ** t)
        v_hat = v / (1 - optimizer_state['beta2'] ** t)
        update = optimizer_state['lr'] * (m_hat / (math.sqrt(v_hat) + optimizer_state['eps']) + optimizer_state['weight_decay'] * data)        
        z= -update
    else:
        z=0        
    
    percentage = normalize_gradient(gradient, scale_min=SCALE_MIN, scale_max=SCALE_MAX, linthresh=LINTHRESH)
    text_radius = (OUTER_RADIUS + INNER_RADIUS) / 2
    log_scale_min, log_scale_max = symlog_scale(SCALE_MIN, LINTHRESH), symlog_scale(SCALE_MAX, LINTHRESH)
            
    START_ANGLE = math.radians(135)
    END_ANGLE = math.radians(45)
    start_x, start_y = angle_to_coords(START_ANGLE, OUTER_RADIUS, CENTER)
    end_x, end_y = angle_to_coords(END_ANGLE, OUTER_RADIUS, CENTER)
    
    knob_angle = math.radians(180 + percentage * 1.8)        
    knob_width, knob_length = RING_WIDTH, RING_WIDTH * 1.1
    knob_x, knob_y = angle_to_coords(knob_angle, OUTER_RADIUS, CENTER)
    knob_end_x, knob_end_y = angle_to_coords(knob_angle, OUTER_RADIUS - knob_length, CENTER)
     

    sqrt_v = math.sqrt(v)
    outer_m_radius = INNER_RADIUS * 0.70 
    inner_v_radius = outer_m_radius * 0.6 
    m_text_radius = (outer_m_radius + inner_v_radius) * .5
        
    m_circle = Circle(cx=CENTER, cy=CENTER, r=outer_m_radius, fill="none", stroke="#888", stroke_width=1, mask=f"url(#top-half-mask)")    
    v_circle = Circle(cx=CENTER, cy=CENTER, r=inner_v_radius, fill="none", stroke="#888", stroke_width=1, mask=f"url(#top-half-mask)")
    
    m_ticks = []
    m_labels = []
    m_values = [-1.00, -0.20, -0.02, 0.00, 0.02, 0.20, 1.00]
    
    label_distance = 0.7
    for i, value in enumerate(m_values):
        angle = math.pi + (i / (len(m_values) - 1)) * math.pi
        tick_start = angle_to_coords(angle, outer_m_radius, CENTER)
        tick_end = angle_to_coords(angle, outer_m_radius + 5, CENTER)
        m_ticks.append(Line(x1=tick_start[0], y1=tick_start[1], x2=tick_end[0], y2=tick_end[1], stroke="#888", stroke_width=.8))                
        text_width = len(f"{value:.2f}") * SIZE / 40 
                
        if angle > math.pi:  # Left side
            label_radius = outer_m_radius + label_distance + text_width / 2
        else:  # Right side
            label_radius = outer_m_radius + label_distance + text_width / 2
        
        label_pos = angle_to_coords(angle, label_radius, CENTER)
                
        rotation = (angle - math.pi) * 180 / math.pi
        if rotation > 90 or rotation < -90:
            rotation += 180
        
        m_labels.append(
            Text(
                f"{value:.2f}",
                x=label_pos[0], y=label_pos[1],
                font_size=SIZE/40,
                fill="#888",
                text_anchor="middle",
                dominant_baseline="central",
                transform=f"rotate({rotation} {label_pos[0]} {label_pos[1]})"
            )
        )
    
    m_angle = pos_to_angle(m, log_scale_min, log_scale_max, LINTHRESH)    
    
    v_ticks = []    
    num_ticks = 20

    for i in range(-num_ticks, num_ticks + 1):        
        angle = m_angle + (i / num_ticks) * (math.pi / 2)    
        angle = angle % (2 * math.pi)
        if angle < math.pi:
            angle += math.pi

        tick_start = angle_to_coords(angle, inner_v_radius, CENTER)
        tick_end = angle_to_coords(angle, inner_v_radius - 5, CENTER)
               
        v_ticks.append(Line(x1=tick_start[0], y1=tick_start[1], x2=tick_end[0], y2=tick_end[1], 
                            stroke="#888", stroke_width=1))

    

    buffer_angle = math.radians(12)
    text_arc_length = math.radians(40)
    if m >= 0:
        end_angle = m_angle - buffer_angle
        start_angle = end_angle - text_arc_length
    else:
        start_angle = m_angle + buffer_angle
        end_angle = start_angle + text_arc_length

    outer_bean_radius = m_text_radius * 1.15
    inner_bean_radius = m_text_radius * 0.85    

    m_bean_group = create_m_bean_with_text(
        param_name,
        m,
        SIZE,
        start_angle,
        end_angle,
        outer_bean_radius,
        inner_bean_radius
    )

    if m == 0:
        m_bean_group = None

    m_indicator = Line(
        x1=CENTER,
        y1=CENTER,
        x2=angle_to_coords(m_angle, outer_m_radius * 1.15, CENTER)[0],
        y2=angle_to_coords(m_angle, outer_m_radius * 1.15, CENTER)[1],
        stroke="#372572", stroke_width=2
    )
    center_circle = Circle(
    cx=SIZE/2,
    cy=SIZE/2,
    r=3,
    fill="#372572"
)
        
    arrow_length = abs(gradient) * (1 - optimizer_state['beta1']) * (OUTER_RADIUS - INNER_RADIUS)        
    max_arrow_length = (OUTER_RADIUS - INNER_RADIUS) * 0.5
    arrow_length = min(arrow_length, max_arrow_length)
            
    max_gradient = 5.0
    blue_arrow = create_arrow_path(CENTER, CENTER, gradient, max_gradient, m_angle, INNER_RADIUS, SIZE)
    
    sqrt_v_shape = create_sqrt_v_visualization(
        param_name=param_name,
        center_x=CENTER,
        center_y=CENTER,
        radius=INNER_RADIUS,
        current_v=v,        
        size=SIZE
    )

    gradient_key = 'negative' if gradient < 0 else 'positive' if gradient > 0 else 'zeroed'
    gradient_id = f"dial-gradient-{gradient_key}"


    mask_path = Path(fill="black").M(SIZE/2, SIZE/2).L(start_x, start_y).A(OUTER_RADIUS, OUTER_RADIUS, 0, 1, 0, end_x, end_y).Z()

    knob_path = Path(fill="white").M(knob_x, knob_y)
    knob_path.L(knob_end_x + knob_width/2 * math.sin(knob_angle), knob_end_y - knob_width/2 * math.cos(knob_angle))
    knob_path.L(knob_end_x - knob_width/2 * math.sin(knob_angle), knob_end_y + knob_width/2 * math.cos(knob_angle))
    knob_path.Z()
    
    z_color = "#4CAF50" if z > 0 else "#F44336"
    z_triangle = "▲" if z > 0 else "▼"    
    rect_width, rect_height = SIZE * 0.4, SIZE * 0.07
    rect_x = SIZE / 2 - rect_width / 2
    rect_y = SIZE / 2 + INNER_RADIUS / 2.4

    z_background = Rect(
        x=rect_x, y=rect_y,
        width=rect_width, height=rect_height,
        rx=rect_height / 2,
        fill=z_color, opacity=0.8
    )

    z_text = Text(
        f"{z_triangle} {abs(z):.6f}",
        x=SIZE/2, y=rect_y + rect_height/2,
        font_family="Arial, sans-serif", font_size=SIZE/18, font_weight="bold",
        text_anchor="middle", dominant_baseline="central", fill="white"
    )
   
    data_text = Text(
        f"{data:.4f}", x=SIZE/2, y=SIZE/2 + INNER_RADIUS/4,
        font_family="Arial, sans-serif", font_size=SIZE/8, font_weight="bold",
        text_anchor="middle", dominant_baseline="central", fill="black"
    )

    text_path = Path(id="textPath", stroke="purple", stroke_width=2, fill="none")
    text_path.M(*angle_to_coords(math.radians(135), text_radius, CENTER))
    text_path.A(text_radius, text_radius, 0, 0, 0, *angle_to_coords(math.radians(45), text_radius, CENTER))

    circle = partial(Circle, cx=CENTER, cy=CENTER)

    return Svg(
        setup_dial_gradients(),
        Defs(
                                 
            Mask(Rect(x=0, y=0, width=SIZE, height=SIZE, fill="white"), mask_path, id="dialMask"),
            Mask(Rect(x=0, y=0, width=SIZE, height=SIZE, fill="white"),
                 Rect(x=0, y=CENTER, width=SIZE, height=CENTER, fill="black"),
                 id="top-half-mask"
                 ),
            Filter(FeDropShadow(dx="4", dy="4", stdDeviation="3", flood_opacity="0.3"), id="dropShadow",),
            text_path, 
            
            
                     
        ),
        circle(r=OUTER_RADIUS, fill=f"url(#{gradient_id})", mask="url(#dialMask)"),
        G(
            circle(r=INNER_RADIUS, fill="white", stroke="none"),
            m_circle, v_circle, knob_path, *m_ticks, *m_labels, *v_ticks,            
            blue_arrow,
            data_text, z_background, z_text,                                                      
            m_indicator, center_circle, sqrt_v_shape, 
            m_bean_group,
            filter="url(#dropShadow)"
        ),        
       
        Text(
            TextPath(
                Tspan(f"{gradient:.4f}", dy="0.4em"),                
                href="#textPath", startOffset="50%",
            ),
            font_family="Arial, sans-serif", font_size=SIZE/10, text_anchor="middle", fill="black"
        ),
        id=param_name,
        width=SIZE, height=SIZE,
        viewBox=f"0 0 {SIZE} {SIZE}",                
    )     

def create_sqrt_v_visualization(param_name: str, center_x: float, center_y: float, radius: float, 
                                current_v: float, size: float):    
    sqrt_v_width = size * 0.20 
    sqrt_v_height = size * 0.10
    sqrt_v_x = center_x - sqrt_v_width/2
    sqrt_v_y = center_y - size/30 

    max_height = sqrt_v_height * 2.5
    current_height = min(max_height * math.sqrt(current_v) / math.sqrt(3), max_height)

    gradients = create_cylinder_gradients(param_name)
    
    cylinder_top = Ellipse(
        sqrt_v_width/2, sqrt_v_height/2,
        cx=center_x, cy=sqrt_v_y,
        fill=f"url(#{param_name}-top-gradient)",
        stroke="none",
    )
    
    side_effect = (Path(fill=f"url(#{param_name}-side-gradient)", stroke="none")
        .M(sqrt_v_x, sqrt_v_y)
        .Q(center_x, sqrt_v_y + current_height * 1.2, sqrt_v_x + sqrt_v_width, sqrt_v_y)
        .L(sqrt_v_x + sqrt_v_width, sqrt_v_y + current_height)
        .Q(center_x, sqrt_v_y + current_height * 0.8, sqrt_v_x, sqrt_v_y + current_height)
        .Z())
    
    cylinder_bottom = Ellipse(
        sqrt_v_width/2, sqrt_v_height/4,
        cx=center_x, cy=sqrt_v_y + current_height,
        fill=f"url(#{param_name}-bottom-gradient)",
        stroke="none",
    )
    
    
    shadow_height = size * 0.04
    shadow = Rect(
        x=sqrt_v_x, 
        y=sqrt_v_y + current_height,
        width=sqrt_v_width,
        height=shadow_height,
        fill=f"url(#{param_name}-shadow-gradient)",
        stroke="none",
    )
    
    sqrt_v_text = Text(
        f"{math.sqrt(current_v):.4f}",
        x=center_x, y=sqrt_v_y + current_height * 0.8,
        font_family="Arial, sans-serif",
        font_size=size/25,
        font_weight="bold",
        text_anchor="middle",
        dominant_baseline="central",
        fill="#FFFFFF"
    )

    return G(
        gradients,
        create_shadow_gradient(param_name),
        shadow,
        side_effect,
        cylinder_bottom,
        cylinder_top,
        sqrt_v_text,
        id=f"{param_name}-sqrt-v-visualization"
    )

def create_shadow_gradient(param_name: str):
    return LinearGradient(
        Stop(offset="0%", stop_color="rgba(0,0,0,0.3)"),
        Stop(offset="100%", stop_color="rgba(0,0,0,0)"),
        id=f"{param_name}-shadow-gradient",
        x1="0%", y1="0%", x2="0%", y2="100%"
    )

def create_cylinder_gradients(param_name: str):
    top_color = "#4a90e2"
    bottom_color = "#2171cd"
    
    top_gradient = RadialGradient(
        Stop(offset="0%", stop_color="white"),
        Stop(offset="70%", stop_color=top_color),
        Stop(offset="100%", stop_color=bottom_color),
        id=f"{param_name}-top-gradient",
        cx="50%", cy="50%", r="50%", fx="25%", fy="25%"
    )
    
    side_gradient = LinearGradient(
        Stop(offset="0%", stop_color=top_color),
        Stop(offset="100%", stop_color=bottom_color),
        id=f"{param_name}-side-gradient",
        x1="0%", y1="0%", x2="0%", y2="100%"
    )
    
    bottom_gradient = RadialGradient(
        Stop(offset="0%", stop_color=bottom_color),
        Stop(offset="100%", stop_color=top_color),
        id=f"{param_name}-bottom-gradient",
        cx="50%", cy="50%", r="50%", fx="25%", fy="25%"
    )
    
    return G(top_gradient, side_gradient, bottom_gradient)


def setup_dial_gradients():
    gradient_defs = {
        'negative': ("#240b36", "#4a1042", "#711845", "#981f3c", "#b72435", "#c31432"),
        'zeroed': ("#fdc830", "#fdc130", "#fcb130", "#fba130", "#f99130", "#f37335"),
        'positive': ("#11998e", "#1eac8e", "#2aba8e", "#35c78d", "#37d18b", "#38db89", "#38ef7d")
    }

    return Defs(*[
        LinearGradient(
            *[Stop(offset=f"{i/(len(colors)-1)*100}%", 
                   style=f"stop-color:{color};stop-opacity:1")
              for i, color in enumerate(colors)],
            id=f"dial-gradient-{key}", x1="0%", y1="0%", x2="100%", y2="100%",
        ) for key, colors in gradient_defs.items()
    ])

def FeDropShadow(dx=0, dy=0, stdDeviation=0, flood_color=None, flood_opacity=None, **kwargs):
    attributes = {
        'dx': dx,
        'dy': dy,
        'stdDeviation': stdDeviation
    }
    if flood_color is not None:
        attributes['flood-color'] = flood_color
    if flood_opacity is not None:
        attributes['flood-opacity'] = flood_opacity
    attributes.update(kwargs)
    return ft_hx('feDropShadow', **attributes)

def symlog_scale(x, linthresh):
    return math.copysign(math.log1p(abs(x) / linthresh), x)

def inv_symlog_scale(y, linthresh):
    return math.copysign(linthresh * (math.exp(abs(y)) - 1), y)

def normalize_gradient(gradient, scale_min, scale_max, linthresh):
    log_gradient = symlog_scale(gradient, linthresh)
    log_min = symlog_scale(scale_min, linthresh)
    log_max = symlog_scale(scale_max, linthresh)
    return (log_gradient - log_min) / (log_max - log_min) * 100

def pos_to_angle(pos, log_scale_min, log_scale_max, linthresh):
    return math.pi + (symlog_scale(pos, linthresh) - log_scale_min) / (log_scale_max - log_scale_min) * math.pi

def angle_to_coords(angle, radius, center):
    return (center + radius * math.cos(angle), center + radius * math.sin(angle))

def create_bean_shape(center_x, center_y, outer_radius, inner_radius, start_angle, end_angle):
        path = Path(fill="#372572", opacity=0.8)                
        path.M(center_x + outer_radius * math.cos(start_angle), 
               center_y + outer_radius * math.sin(start_angle))
        path.A(outer_radius, outer_radius, 0, 0, 1, 
               center_x + outer_radius * math.cos(end_angle), 
               center_y + outer_radius * math.sin(end_angle))                
        edge_radius = (outer_radius - inner_radius) / 2
        path.A(edge_radius, edge_radius, 0, 0, 1,
               center_x + inner_radius * math.cos(end_angle),
               center_y + inner_radius * math.sin(end_angle))                
        path.A(inner_radius, inner_radius, 0, 0, 0,
               center_x + inner_radius * math.cos(start_angle),
               center_y + inner_radius * math.sin(start_angle))                
        path.A(edge_radius, edge_radius, 0, 0, 1,
               center_x + outer_radius * math.cos(start_angle),
               center_y + outer_radius * math.sin(start_angle))        
        path.Z()
        return path

def create_convex_shape(x, y, width, height):
    path = Path(fill="#4a90e2", opacity=0.8)       
    ctrl_offset = height * 1.2       
    path.M(x, y + height)    
    path.Q(x + width/2, y + height + ctrl_offset, x + width, y + height)    
    path.L(x + width, y)        
    path.Q(x + width/2, y - ctrl_offset, x, y)        
    path.L(x, y + height)    
    path.Z()
    return path


def create_sqrtv_shape_with_cutout_text(shape, text, font_size, param_name, text_path_id=None):
    mask_id = f"sqrtv-mask-{param_name}"
    
    if text_path_id:
        text_element = Text(
            TextPath(
                text,
                href=f"#{text_path_id}",
                startOffset="50%"
            ),
            font_family="Arial, sans-serif",
            font_size=font_size,
            font_weight="bold",
            fill="black",
            text_anchor="middle",
            dominant_baseline="central"
        )
    else:
        text_element = Text(
            text,
            x=shape.x + shape.width/2,
            y=shape.y + shape.height/2,
            font_family="Arial, sans-serif", 
            font_size=font_size, 
            font_weight="bold",
            text_anchor="middle", 
            dominant_baseline="central", 
            fill="black"
        )

    mask = Mask(
        Rect(x=0, y=0, width="100%", height="100%", fill="white"), 
        text_element,
        id=mask_id
    )

    shape.mask = f"url(#{mask_id})"

    return shape, mask

def create_m_bean_with_text(param_name, m, size, start_angle, end_angle, outer_radius, inner_radius):
    center = size / 2
    text_radius = (outer_radius + inner_radius) / 2
    
    bean_shape = create_bean_shape(center, center, outer_radius, inner_radius, start_angle, end_angle)
    bean_shape.fill = "#372572"  # Purple color
    
    text_path = Path(id=f"{param_name}-m-text-path", fill="none")
    text_path.M(center + text_radius * math.cos(start_angle), 
                center + text_radius * math.sin(start_angle))
    text_path.A(text_radius, text_radius, 0, 0, 1, 
                center + text_radius * math.cos(end_angle), 
                center + text_radius * math.sin(end_angle))
    
    text_element = Text(
        TextPath(
            f"{m:.4f}",
            href=f"#{param_name}-m-text-path",
            startOffset="50%"
        ),
        font_family="Arial, sans-serif",
        font_size=size/30,
        font_weight="bold",
        fill="white",
        text_anchor="middle",
        dominant_baseline="central"
    )
    
    return G(bean_shape, text_path, text_element)

def create_arrow_path(center_x, center_y, gradient, max_gradient, m_angle, inner_radius, size):
    arc_radius = inner_radius * 0.76
    start = (
        center_x + arc_radius * math.cos(m_angle),
        center_y + arc_radius * math.sin(m_angle)
    )
        
    min_length = math.radians(5)  # Minimum 5 degrees for visibility
    max_length = math.pi / 2  # Maximum 90 degrees
    gradient_length = max(min(abs(gradient) / max_gradient * max_length, max_length), min_length)
    gradient_length *= math.copysign(1, gradient)
    
    end_angle = m_angle + gradient_length
    end = (
        center_x + arc_radius * math.cos(end_angle),
        center_y + arc_radius * math.sin(end_angle)
    )
    
    if gradient == 0:
        return None    
    arrow_path = Path(stroke="blue", stroke_width=2, fill="blue")
    arrow_path.M(*start)
    arrow_path.A(arc_radius, arc_radius, 0, 0, int(gradient > 0), *end)
    base_arrow_length = size / 30
    base_arrow_width = size / 90        
    scale_factor = 0.5 + 0.5 * min(abs(gradient) / max_gradient, 1)
    arrow_length = base_arrow_length * scale_factor
    arrow_width = base_arrow_width * scale_factor        
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    cos_angle, sin_angle = math.cos(angle), math.sin(angle)        
    arrowhead = [
        end,
        (end[0] - arrow_length * cos_angle + arrow_width * sin_angle,
         end[1] - arrow_length * sin_angle - arrow_width * cos_angle),
        (end[0] - arrow_length * cos_angle - arrow_width * sin_angle,
         end[1] - arrow_length * sin_angle + arrow_width * cos_angle)
    ]
        
    arrow_path.M(*arrowhead[0])
    arrow_path.L(*arrowhead[1])
    arrow_path.L(*arrowhead[2])
    arrow_path.Z()
    
    return arrow_path



# -------------------------------------------------------------------------------------------------
# Parameter Dial Testing
# -------------------------------------------------------------------------------------------------

@rt('/parameter_dial')
def get():
    random_data = random.uniform(-1, 1)
    random_m = random.uniform(-0.02, 0.02)
    random_v = random.uniform(0.003, 0.03)
    random_t = random.uniform(1, 100)
    optimizer_state = {
        'lr': 0.1,
        'beta1': 0.9,
        'beta2': 0.999,
        'eps': 1e-8,
        'weight_decay': 1e-4,
        't': random_t
    }   
    return Div(
        Div(
            Span("Select Gradient: ", cls="mr-2 font-bold"),
            Div(
                *[Label(
                    Input(type="radio", name="gradient", value=value, 
                          hx_post="/update-dial", hx_trigger="change", hx_target="#parameter-dial", hx_swap="outerHTML",
                          checked=(value == "0.0000"), cls="radio-input"),
                    Span(cls="radio-circle"),
                    label,
                    cls="radio-item"
                ) for value, label in [("-3.0000", "Low (<0)"), ("0.0000", "Mid (0)"), ("1.0000", "High (>0)")]], 
                cls="radio-group"
            ),
            cls="mt-1 mb-4"
        ),
        create_parameter_dial(param_name="parameter-dial", data=random_data, gradient=0.0000, m=random_m, v=random_v, optimizer_state=optimizer_state),
        cls="p-6 rounded-lg shadow-md mx-auto min-h-screen bg-[#e0e8d8] text-black"
    )

@rt("/update-dial")
async def update_dial(request: Request):
    form = await request.form()
    gradient = float(form.get("gradient", 0))
        
    if gradient < 0:  # Low
        m = random.uniform(-0.5, -0.1)
        v = random.uniform(0.01, 0.1)
    elif gradient > 0:  # High
        m = random.uniform(0.1, 0.5)
        v = random.uniform(0.01, 0.1)
    else:  # Mid
        m = random.uniform(-0.1, 0.1)
        v = random.uniform(0.001, 0.01)
    
    random_data = random.uniform(-1, 1)
    optimizer_state = {
        'lr': 0.1,
        'beta1': 0.9,
        'beta2': 0.999,
        'eps': 1e-8,
        'weight_decay': 1e-4,
        't': random.uniform(0, 100) 
    }
    return create_parameter_dial(param_name="parameter-dial", data=random_data, gradient=gradient, m=m, v=v, optimizer_state=optimizer_state)


# Run the app
serve()
