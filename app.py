import base64
import httpx
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
app, rt = fast_app(
    pico=False,    
    hdrs=(tailwindLink, sselink, chartlink)
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
        self.initialize_model()

    def initialize_model(self):
        self.model = MLP(2, [8, 3])
        self.optimizer = AdamW(self.model.parameters(), lr=1e-1, weight_decay=1e-4)

    def reset(self):
        if self.model is None:
            self.initialize_model()
        
        # Reset training progress
        self.is_training = False
        self.step_count = 0
        self.train_losses = []
        self.val_losses = []
        
        # Reset model parameters to initial values
        for p in self.model.parameters():
            p.data = random.uniform(-0.1, 0.1)
        
        # Reset optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=1e-1, weight_decay=1e-4)

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
            self.optimizer.zero_grad()

            self.step_count += 1
            self.train_losses.append(loss.data)

            # Evaluate validation loss every 10 steps
            if self.step_count % 10 == 0 and self.val_split:
                X_val, y_val = zip(*self.val_split)
                val_loss = self.loss_fun(X_val, y_val)
                self.val_losses.append(val_loss.data)
        
        return self.get_current_step()

    def get_current_step(self):
        return {
            'step_count': self.step_count,
            'train_loss': f"{self.train_losses[-1]:.6f}" if self.train_losses else "---",
            'val_loss': f"{self.val_losses[-1]:.6f}" if self.val_losses and len(self.val_losses) > 0 else "---",
            'is_training': self.is_training
        }

    def loss_fun(self, X, y):
        total_loss = Value(0.0)
        for x, y in zip(X, y):
            logits = self.model(x)
            loss = cross_entropy(logits, y)
            total_loss = total_loss + loss
        mean_loss = total_loss * (1.0 / len(X))
        return mean_loss

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

animation_script = """
let isTraining = false;
let eventSource;
let reconnectAttempts = 0;
const maxReconnectAttempts = 5;
const reconnectDelay = 5000; // 5 seconds

function updatePlayPauseButton() {
    const playPauseBtn = document.getElementById('play-pause-btn');
    if (isTraining) {
        playPauseBtn.textContent = 'Pause';
        playPauseBtn.classList.remove('bg-green-500');
        playPauseBtn.classList.add('bg-red-500');
    } else {
        playPauseBtn.textContent = 'Play';
        playPauseBtn.classList.remove('bg-red-500');
        playPauseBtn.classList.add('bg-green-500');
    }
}

function updateSSEConnection() {
    if (eventSource) {
        eventSource.close();
    }
    
    if (reconnectAttempts >= maxReconnectAttempts) {
        console.error('Max reconnection attempts reached. Please refresh the page.');
        return;
    }

    const url = `/train_stream?train=${isTraining}`;
    eventSource = new EventSource(url);
    
    eventSource.addEventListener('open', function(e) {
        console.log('SSE connection opened');
        reconnectAttempts = 0; // Reset attempts on successful connection
    });

    eventSource.addEventListener('step', function(e) {
        const data = JSON.parse(e.data);
        updateProgressTracker(data);
    });

    eventSource.addEventListener('heartbeat', function(e) {
        console.log('Heartbeat received');
    });

    eventSource.addEventListener('error', function(e) {
        console.error('SSE connection error:', e);
        eventSource.close();
        reconnectAttempts++;
        setTimeout(updateSSEConnection, reconnectDelay);
    });
}

function handlePlayPause() {
    isTraining = !isTraining;
    updatePlayPauseButton();
    updateSSEConnection();
}

function handleTrainStep() {
    if (!isTraining) {
        fetch('/train_stream?step=true')
            .then(response => response.text())
            .then(text => {
                const lines = text.split('\\n');
                const eventData = lines.find(line => line.startsWith('data:'));
                if (eventData) {
                    const data = JSON.parse(eventData.slice(5));
                    updateProgressTracker(data);
                }
            })
            .catch(error => console.error('Error during single step:', error));
    }
}

function handleReset() {
    fetch('/reset', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            updateProgressTracker(data);
            isTraining = false;
            updatePlayPauseButton();
            updateSSEConnection();
            console.log(`Reset to step 0`);
            resetChart();
        });
            
}

function updateProgressTracker(data) {
    const stepText = document.getElementById('step-text');
    const trainLossText = document.getElementById('train-loss-text');
    const valLossText = document.getElementById('val-loss-text');
    const chartContainer = document.getElementById('loss-chart-container');

    if (stepText) stepText.textContent = `Step ${data.step_count}/100`;
    if (trainLossText) {
        trainLossText.textContent = `Train loss: ${data.train_loss}`;
        trainLossText.dataset.value = data.train_loss;
    }
    if (valLossText) {
        valLossText.textContent = `Validation loss: ${data.val_loss}`;
        valLossText.dataset.value = data.val_loss;
    }

    // Show/hide chart based on step count
    if (data.step_count > 0) {
        chartContainer.classList.remove('hidden');
        updateLossChart(data.step_count, parseFloat(data.train_loss), parseFloat(data.val_loss));
    } else {
        chartContainer.classList.add('hidden');
    }

    // Trigger a custom event
    const event = new CustomEvent('progressUpdated', { detail: data });
    document.dispatchEvent(event);
}

// Initialize loss chart
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

// Listen for SSE events
document.body.addEventListener('htmx:sseMessage', function(evt) {
    if (evt.detail.type === 'step') {
        const data = JSON.parse(evt.detail.data);                
        updateProgressTracker(data);
    }
});
        
       

// Add event listeners
document.getElementById('play-pause-btn').addEventListener('click', handlePlayPause);
document.getElementById('step-btn').addEventListener('click', handleTrainStep);
document.getElementById('reset-btn').addEventListener('click', handleReset);


window.addEventListener('beforeunload', function() {
    if (eventSource) {
        eventSource.close();
    }
});

// Initialize SSE connection when the page loads
document.addEventListener('DOMContentLoaded', function() {
    updateSSEConnection();
    initChart();
});
"""


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



def create_graph_svg(graph_data):
    svg_content = []
    
    # Create nodes
    for index, node in enumerate(graph_data['nodes']):
        x = 50 + (index % 5) * 100
        y = 50 + (index // 5) * 100
        
        g = G(
            Circle(r=20, fill='white', stroke='black'),
            Text(node['value'].toFixed(4), 
                 text_anchor='middle', 
                 dominant_baseline='middle'),
            id=f"node-{node['id']}",
            transform=f"translate({x}, {y})"
        )
        svg_content.append(g)
    
    # Create edges
    for edge in graph_data['edges']:
        from_node = f"#node-{edge['from']}"
        to_node = f"#node-{edge['to']}"
        line = Line(x1=0, y1=0, x2=0, y2=0,  # These will be set by JavaScript
                    stroke='black',
                    marker_end='url(#arrowhead)')
        svg_content.append(line)
    
    return Svg(*svg_content, 
               width="100%", height="100%", viewBox="0 0 600 400")


@rt('/get_graph')
async def get():
    graph_data = visualizer.get_graph_data()
    return create_graph_svg(graph_data)

@rt('/get_dataset_viz')
async def get():    
    return Div("Dataset visualization placeholder")

@rt('/reset')
async def post():
    visualizer.reset()
    return {
        'step_count': 0,
        'train_loss': "---",
        'val_loss': "---"
    }


@rt('/adamw')
async def get():    
    return Titled("AdamW Optimizer Explainer",
        Style("""
            body { font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; max-width: 800px; margin: 0 auto; }
            h1, h2 { color: #333; }
            pre { background-color: #f4f4f4; padding: 10px; border-radius: 5px; }
            .math { font-style: italic; }
        """),
        Link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css", integrity="sha384-nB0miv6/jRmo5UMMR1wu3Gz6NLsoTkbqJghGIsx//Rlm+ZU03BU6SQNC66uf4l5+", crossorigin="anonymous"),
        Script(defer=True, src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js", integrity="sha384-7zkQWkzuo3B5mTepMUcHkMB5jZaolc2xDwL6VFqjFALcbeS9Ggm/Yr2r3Dy4lfFg", crossorigin="anonymous"),
        # Script(defer=True, src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js", integrity="sha384-43gviWU0YVjaDtb/GhzOouOXtZMP/7XUzwPTstBeZFe/+rCMvRwr4yROQP43s0Xk", crossorigin="anonymous", onload="renderMathInElement(document.body);"),
        H2("1. Initialization"),
        P("AdamW is initialized with the following parameters:"),
        Ul(
            Li(Span("lr", cls="math"), ": learning rate (default: 0.1)"),
            Li(Span("β₁, β₂", cls="math"), ": exponential decay rates for moment estimates (default: 0.9, 0.95)"),
            Li(Span("ε", cls="math"), ": small constant for numerical stability (default: 1e-8)"),
            Li(Span("weight_decay", cls="math"), ": weight decay factor (default: 0.0)")
        ),
        P("For each parameter ", Span("p", cls="math"), ", we initialize:"),
        Ul(
            Li(Span("m = 0", cls="math"), " (first moment vector)"),
            Li(Span("v = 0", cls="math"), " (second moment vector)")
        ),
        H2("2. Update Step"),
        P("For each parameter ", Span("p", cls="math"), " and its gradient ", Span("g", cls="math"), ":"),
        Ol(
            Li("Update biased first moment estimate:", Br(),
               Span("m = β₁ * m + (1 - β₁) * g", cls="math")),
            Li("Update biased second raw moment estimate:", Br(),
               Span("v = β₂ * v + (1 - β₂) * g²", cls="math")),
            Li("Correct bias in first moment:", Br(),
               Span("m̂ = m / (1 - β₁ᵗ)", cls="math")),
            Li("Correct bias in second moment:", Br(),
               Span("v̂ = v / (1 - β₂ᵗ)", cls="math")),
            Li("Compute the parameter update:", Br(),
               Span("Δp = -lr * (m̂ / (√v̂ + ε) + weight_decay * p)", cls="math")),
            Li("Apply the update:", Br(),
               Span("p = p + Δp", cls="math"))
        ),
        H2("3. Key Concepts"),
        Ul(
            Li(Strong("Adaptive Learning Rate"), ": The effective learning rate is adjusted based on the first and second moments of the gradients."),
            Li(Strong("Momentum"), ": The first moment (m) provides a form of momentum, helping to smooth out updates."),
            Li(Strong("RMSprop-like Scaling"), ": The second moment (v) scales the update, similar to RMSprop, helping to handle different magnitudes of gradients."),
            Li(Strong("Bias Correction"), ": The bias correction terms (1 - β₁ᵗ and 1 - β₂ᵗ) help to reduce bias in the early steps of training."),
            Li(Strong("Weight Decay"), ": Unlike standard Adam, AdamW applies weight decay directly to the parameters, which can improve generalization.")
        ),
        H2("4. Visualization"),
        P("The optimizer state table in the main view shows:"),
        Ul(
            Li(Strong("param"), ": Current parameter value"),
            Li(Strong("-m/sqrt(v)"), ": The main component of the update (before applying learning rate)"),
            Li(Strong("grad"), ": Current gradient"),
            Li(Strong("m"), ": First moment estimate"),
            Li(Strong("sqrt(v)"), ": Square root of the second moment estimate")
        ),
        P("Green values are positive, red are negative, helping to visualize the direction of updates and gradient flow.")                 
    )


@rt('/model_info')
def get(request: Request):
    page = int(request.query_params.get('page', '1'))
    model = visualizer.model
    
    # Calculate total number of parameters
    total_params = sum(len(n.w) + 1 for layer in model.layers for n in layer.neurons)
    
    # Define page titles
    page_titles = {
        1: "Neural Network Structure",
        2: "Parameter Dials"
    }
    
    descriptions = {
        1: {
            "above": "This diagram shows the structure of our neural network. Each circle represents a neuron, and the lines represent connections between neurons.",
            "below": "The network consists of an input layer, hidden layers, and an output layer. The number of neurons in each layer determines the network's capacity to learn complex patterns."
        },
        2: {
            "above": f"This diagram shows the structure of the neural network. Total learnable parameters: {total_params}",
            "below": "The position of each dial indicates the current value of the parameter. As training progresses, these dials will rotate to optimize the network's predictions."
        },
        3: {
            "above": "This visualization shows the activation levels of neurons in our network. The brightness of each neuron indicates its level of activation.",
            "below": "Observing these activations can help us understand which parts of the network are most responsive to different inputs, providing insights into the network's decision-making process."
        }
    }
    # Get the appropriate SVG content based on the page
    if page == 1:
        svg_content = create_network_structure(model)
        svg_element = Svg(*svg_content, 
            width="100%", height="100%", 
            viewBox="0 0 800 400",
            preserveAspectRatio="xMidYMid meet",
            cls="w-full aspect-[2/1] max-w-[800px] mx-auto sample-transition"),
        svg_container = Div(svg_element, cls="w-full aspect-[2/1]") 
    elif page == 2:
        svg_content = create_parameter_dials(model)
        svg_element = NotStr(svg_content) 
        svg_container = Div(svg_element, cls="w-full aspect-[2/1]") 
    else:
        svg_container = Div() 

    return Div(
        H2(page_titles.get(page, "Unknown Page"), cls="text-2xl font-bold mb-4"),
        P(descriptions[page]["above"], cls="text-lg mb-4"),
        svg_container,
        P(descriptions[page]["below"], cls="text-lg mt-4"),        
        Div(
            Button("Prev", 
                   hx_get=f"/model_info?page={page-1}",
                   hx_target="#model-info-container",
                   hx_swap="innerHTML transition:true",
                   cls="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded",
                   disabled=page == 1),
            Button("Next", 
                   hx_get=f"/model_info?page={page+1}",
                   hx_target="#model-info-container",
                   hx_swap="innerHTML transition:true",
                   cls="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded",
                   disabled=page == 2),
            cls="flex justify-between mt-4"
        ),        
        id="model-info-container",
        cls="flex flex-col w-full max-w-4xl mx-auto p-4"
    )

def create_network_structure(model):
    width = 800
    height = 400
    input_x = 50
    output_x = width - 50
    
    svg_content = []
    
    input_size = len(model.layers[0].neurons[0].w)
    hidden_layers = model.layers[:-1]
    output_size = len(model.layers[-1].neurons)
    layer_width = (output_x - input_x) / (len(hidden_layers) + 1)
    
    # Create edges first
    for i in range(len(model.layers)):
        start_x = input_x + i * layer_width
        end_x = input_x + (i + 1) * layer_width
        start_neurons = len(model.layers[i-1].neurons) if i > 0 else input_size
        end_neurons = len(model.layers[i].neurons)
        
        for j in range(start_neurons):
            for k in range(end_neurons):
                y1 = (j + 1) * (height / (start_neurons + 1))
                y2 = (k + 1) * (height / (end_neurons + 1))
                svg_content.append(Line(x1=start_x, y1=y1, x2=end_x, y2=y2, stroke='#ccc'))
    
    # Create input layer shapes
    for i in range(input_size):
        y = (i + 1) * (height / (input_size + 1))
        rect = Rect(width=40, height=40, x=-20, y=-20, fill='#6ab7ff')
        g = G(rect, transform=f'translate({input_x}, {y})')
        svg_content.append(g)
    
    # Create hidden layer shapes
    for i, layer in enumerate(hidden_layers):
        x = input_x + (i + 1) * layer_width
        for j, neuron in enumerate(layer.neurons):
            y = (j + 1) * (height / (len(layer.neurons) + 1))
            circle = Circle(r=20, fill='#6aff9e')
            g = G(circle, transform=f'translate({x}, {y})')
            svg_content.append(g)
    
    # Create output layer shapes
    for i in range(output_size):
        y = (i + 1) * (height / (output_size + 1))
        circle = Circle(r=20, fill='#ff9e6a')
        g = G(circle, transform=f'translate({output_x}, {y})')
        svg_content.append(g)
    
    # Create input layer text
    for i in range(input_size):
        y = (i + 1) * (height / (input_size + 1))
        label = f'INPUT{i+1}'
        text_bg = Rect(width=len(label)*7.5, height=18, x=input_x-len(label)*3.75, y=y-9, 
                       fill='white', opacity=0.7, rx=3, ry=3)
        text = Text(label, x=input_x, y=y+5, text_anchor='middle', font_size=12)
        svg_content.append(G(text_bg, text))
    
    # Create hidden layer text
    for i, layer in enumerate(hidden_layers):
        x = input_x + (i + 1) * layer_width
        for j, neuron in enumerate(layer.neurons):
            y = (j + 1) * (height / (len(layer.neurons) + 1))
            label = f'HIDDEN{j+1}'
            text_bg = Rect(width=len(label)*7.5, height=18, x=x-len(label)*3.75, y=y-9, 
                           fill='white', opacity=0.7, rx=3, ry=3)
            text = Text(label, x=x, y=y+5, text_anchor='middle', font_size=12)
            svg_content.append(G(text_bg, text))
    
    # Create output layer text
    for i in range(output_size):
        y = (i + 1) * (height / (output_size + 1))
        label = f'OUTPUT{i+1}'
        text_bg = Rect(width=len(label)*7.5, height=18, x=output_x-len(label)*3.75, y=y-9, 
                       fill='white', opacity=0.7, rx=3, ry=3)
        text = Text(label, x=output_x, y=y+5, text_anchor='middle', font_size=12)
        svg_content.append(G(text_bg, text))
    
    return svg_content

def create_parameter_dials(model):
    svg_content = '<svg width="100%" height="100%" viewBox="0 0 800 400" preserveAspectRatio="xMidYMid meet">'
    dial_size = 40
    dial_gap = 20  # Increased gap between dials
    
    total_dials = sum(len(neuron.w) + 1 for layer in model.layers for neuron in layer.neurons)
    
    # Calculate the number of columns and rows that fit within the SVG
    max_columns = (800 - dial_gap) // (dial_size + dial_gap)
    max_rows = (400 - dial_gap) // (dial_size + dial_gap)
    
    columns = min(max_columns, total_dials)
    rows = min(max_rows, math.ceil(total_dials / columns))
    
    # Recalculate columns if we exceed max_rows
    if rows == max_rows:
        columns = min(max_columns, math.ceil(total_dials / max_rows))
    
    grid_width = columns * (dial_size + dial_gap) - dial_gap
    grid_height = rows * (dial_size + dial_gap) - dial_gap
    
    start_x = (800 - grid_width) / 2
    start_y = (400 - grid_height) / 2
    
    # Add a background rectangle
    svg_content += f'<rect x="0" y="0" width="800" height="400" fill="#f0f0f0"/>'
    
    # Group all dials
    svg_content += f'<g transform="translate({start_x}, {start_y})">'
    
    param_index = 0
    for layer in model.layers:
        for neuron in layer.neurons:
            for w in neuron.w:
                if param_index >= columns * rows:
                    break  # Stop if we've filled all available slots
                
                x = (param_index % columns) * (dial_size + dial_gap)
                y = (param_index // columns) * (dial_size + dial_gap)
                
                normalized_data = max(min(w.data, 1), -1)
                normalized_grad = max(min(w.grad, 1), -1)
                
                svg_content += create_dial(x, y, dial_size, normalized_data, normalized_grad, w.data)
                
                param_index += 1
            
            # Add bias dial
            if param_index < columns * rows:
                x = (param_index % columns) * (dial_size + dial_gap)
                y = (param_index // columns) * (dial_size + dial_gap)
                
                normalized_data = max(min(neuron.b.data, 1), -1)
                normalized_grad = max(min(neuron.b.grad, 1), -1)
                
                svg_content += create_dial(x, y, dial_size, normalized_data, normalized_grad, neuron.b.data)
                
                param_index += 1
    
    svg_content += '</g></svg>'
    return svg_content

def create_dial(x, y, size, normalized_data, normalized_grad, value):
    dial = f'<g transform="translate({x}, {y})">'
    
    # Background circle
    dial += f'<circle cx="{size/2}" cy="{size/2}" r="{size/2}" fill="white" stroke="#d0d0d0" stroke-width="1"/>'
    
    # Gradient arc
    gradient_color = "#4CAF50" if normalized_grad > 0 else "#F44336"
    end_angle = 180 + normalized_grad * 180
    large_arc_flag = 1 if abs(normalized_grad) > 0.5 else 0
    path = f"M{size/2},{size/2} L{size/2},0 A{size/2},{size/2} 0 {large_arc_flag},1 {size/2 + size/2*math.sin(math.radians(end_angle))},{size/2 - size/2*math.cos(math.radians(end_angle))} Z"
    dial += f'<path d="{path}" fill="{gradient_color}" fill-opacity="0.3"/>'
    
    # Data indicator line
    data_angle = 180 + normalized_data * 180
    dial += f'<line x1="{size/2}" y1="{size/2}" x2="{size/2 + size/2*0.8*math.sin(math.radians(data_angle))}" y2="{size/2 - size/2*0.8*math.cos(math.radians(data_angle))}" stroke="black" stroke-width="2"/>'
    
    # Center dot
    dial += f'<circle cx="{size/2}" cy="{size/2}" r="1" fill="black"/>'
    
    # Value text
    dial += f'<text x="{size/2}" y="{size+10}" text-anchor="middle" font-size="{size/5}" fill="#333">{value:.2f}</text>'
    
    dial += '</g>'
    return dial

# ------------------------------------------------------------------------------------------
# ----------------------------------------Dial--------------------------------------------
# ------------------------------------------------------------------------------------------


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

# def normalize_gradient(gradient):
#     if gradient == 0:
#         return 50.0
#     sign = 1 if gradient > 0 else -1
#     return 50 + sign * 50 * (1 - math.exp(-abs(gradient))) / (1 + math.exp(-abs(gradient)))



def symlog_scale(x, linthresh=0.01):
    return math.copysign(math.log1p(abs(x) / linthresh), x)

def inv_symlog_scale(y, linthresh=0.01):
    return math.copysign(linthresh * (math.exp(abs(y)) - 1), y)

def normalize_gradient(gradient, scale_min=-1, scale_max=1, linthresh=0.01):
    log_gradient = symlog_scale(gradient, linthresh)
    log_min = symlog_scale(scale_min, linthresh)
    log_max = symlog_scale(scale_max, linthresh)
    return (log_gradient - log_min) / (log_max - log_min) * 100



def create_parameter_dial(data=1.0000, gradient=25.0000, m=0.1, v=1.0, size=200):    
    std_dev = math.sqrt(v)
    percentage = normalize_gradient(gradient)
    ring_width = size / 8
    outer_radius = size / 2 - 5
    inner_radius = outer_radius - ring_width
    text_radius = (outer_radius + inner_radius) / 2
    center = size / 2
       
    scale_min, scale_max = -1.0, 1.0
    linthresh = 0.01
    log_scale_min, log_scale_max = symlog_scale(scale_min, linthresh), symlog_scale(scale_max, linthresh)
    
    def pos_to_angle(pos):
        return math.pi + (symlog_scale(pos, linthresh) - log_scale_min) / (log_scale_max - log_scale_min) * math.pi
    
    angle_to_coords = lambda angle, radius: (center + radius * math.cos(angle), center + radius * math.sin(angle))

    start_angle, end_angle = math.radians(135), math.radians(45)
    start_x, start_y = angle_to_coords(start_angle, outer_radius)
    end_x, end_y = angle_to_coords(end_angle, outer_radius)
    
    knob_angle = math.radians(180 + percentage * 1.8)        
    knob_width, knob_length = ring_width, ring_width * 1.1
    knob_x, knob_y = angle_to_coords(knob_angle, outer_radius)
    knob_end_x, knob_end_y = angle_to_coords(knob_angle, outer_radius - knob_length)

    # Create the radio dial arc with mask (top semicircle only)    
    radio_dial_arc = G(
        Path(fill="none")
            .M(center-inner_radius, center)
            .A(inner_radius, inner_radius, 0, 1, 1, center+inner_radius, center),
        Circle(r=inner_radius * 0.5, cx=center, cy=center, fill="white"),
        id="radio-dial-arc"
    )

    # Calculate standard deviation area
    std_start = max(scale_min, m - std_dev)
    std_end = min(scale_max, m + std_dev)
    std_area_start_angle = pos_to_angle(std_start)
    std_area_end_angle = pos_to_angle(std_end)

    std_area = Path(fill="rgba(255, 192, 203, 0.5)")
    std_area.M(*angle_to_coords(std_area_start_angle, inner_radius * 0.5))
    std_area.A(inner_radius * 0.5, inner_radius * 0.5, 0, 0, 1, 
               *angle_to_coords(std_area_end_angle, inner_radius * 0.5))
    std_area.L(*angle_to_coords(std_area_end_angle, inner_radius))
    std_area.A(inner_radius, inner_radius, 0, 0, 0, 
               *angle_to_coords(std_area_start_angle, inner_radius))
    std_area.Z()
    
    
    # Create labels with correct positioning
    label_radius = inner_radius * 0.65 
    label_values = [scale_min, scale_min/5, scale_min/50, 0, scale_max/50, scale_max/5, scale_max]
    
    std_dev_labels = [
        Text(
            f"{value:.2f}",
            x=center + label_radius * math.cos(math.pi + (i / (len(label_values) - 1)) * math.pi),
            y=center + label_radius * math.sin(math.pi + (i / (len(label_values) - 1)) * math.pi),
            font_size=size/24,
            text_anchor="middle",
            dominant_baseline="middle"
        )
        for i, value in enumerate(label_values)
    ]

    ticks = [
        Line(
            *angle_to_coords(math.pi + (i / (len(label_values) - 1)) * math.pi, inner_radius),
            *angle_to_coords(math.pi + (i / (len(label_values) - 1)) * math.pi, inner_radius - ring_width/2),
            stroke="black", stroke_width=1, stroke_opacity=0.2
        )
        for i in range(len(label_values))
    ]

    # Update m indicator position
    m_angle = pos_to_angle(m)
    m_indicator = Line(
        *angle_to_coords(m_angle, inner_radius * 0.5),
        *angle_to_coords(m_angle, inner_radius),
        stroke="red", stroke_width=2
    )

    # Create the m text path
    m_text_radius = inner_radius * 0.35
    m_text_path = Path(id="m-value-path", fill="none")
    m_text_path.M(center-m_text_radius, center).A(m_text_radius, m_text_radius, 0, 1, 1, center+m_text_radius, center)


    m_normalized = (symlog_scale(m, linthresh) - log_scale_min) / (log_scale_max - log_scale_min)

    # Adjust the range to prevent cut-off
    text_margin = 0.2
    m_adjusted = text_margin + m_normalized * (1 - 2 * text_margin)
    
    m_value_text = Text(
        TextPath(
            Tspan(f"{m:.4f}"),
            href="#m-value-path",
            startOffset=f"{m_adjusted * 100}%"
        ),
        fill="red", font_size=size/24, font_weight="bold", text_anchor="middle"
    )

    gradient_defs = {
        'low': ("#240b36", "#4a1042", "#711845", "#981f3c", "#b72435", "#c31432"),
        'mid': ("#fdc830", "#fdc130", "#fcb130", "#fba130", "#f99130", "#f37335"),
        'high': ("#11998e", "#1eac8e", "#2aba8e", "#35c78d", "#37d18b", "#38db89", "#38ef7d")
    }

    gradient_key = 'low' if percentage < 50 else 'high' if percentage > 50 else 'mid'
    emoji = {"low": "👇", "mid": "🙅‍♀️", "high": "👆"}[gradient_key]

    gradient_def = LinearGradient(
        *[Stop(offset=f"{i/(len(gradient_defs[gradient_key])-1)*100}%", 
               style=f"stop-color:{color};stop-opacity:1")
          for i, color in enumerate(gradient_defs[gradient_key])],
        id="dialGradient", x1="0%", y1="0%", x2="100%", y2="100%",
    )

    mask_path = Path(fill="black").M(size/2, size/2).L(start_x, start_y).A(outer_radius, outer_radius, 0, 1, 0, end_x, end_y).Z()

    knob_path = Path(fill="white").M(knob_x, knob_y)
    knob_path.L(knob_end_x + knob_width/2 * math.sin(knob_angle), knob_end_y - knob_width/2 * math.cos(knob_angle))
    knob_path.L(knob_end_x - knob_width/2 * math.sin(knob_angle), knob_end_y + knob_width/2 * math.cos(knob_angle))
    knob_path.Z()

    # Calculate z = m/sqrt(v)
    z = m / math.sqrt(v)
    z_color = "#4CAF50" if z > 0 else "#F44336"
    z_triangle = "▲" if z > 0 else "▼"

    # Create rounded rectangle for z-value background
    rect_width, rect_height = size * 0.4, size * 0.07
    rect_x = size / 2 - rect_width / 2
    rect_y = size / 2 + inner_radius / 1.8

    rect_attrs = dict(
        x=rect_x, y=rect_y,
        width=rect_width, height=rect_height,
        rx=rect_height / 2
    )

    z_background = Rect(fill=z_color, **rect_attrs, opacity=0.8)

    z_text = Text(
        f"{z_triangle} {abs(z):.4f}",
        x=size / 2, y=rect_y + rect_height / 2,
        font_family="Arial, sans-serif", font_size=size / 16, font_weight="bold",
        text_anchor="middle", dominant_baseline="central", fill="white"
    )

    # Create a cutout effect
    z_cutout = Rect(fill="white", mask="url(#z-text-mask)", **rect_attrs)

    data_text = Text(
        f"{data:.4f}", x=size/2, y=size/2 + inner_radius/2.5,
        font_family="Arial, sans-serif", font_size=size/8, font_weight="bold",
        text_anchor="middle", dominant_baseline="central", fill="black"
    )

    text_path = Path(id="textPath", stroke="purple", stroke_width=2, fill="none")
    text_path.M(*angle_to_coords(math.radians(135), text_radius))
    text_path.A(text_radius, text_radius, 0, 0, 0, *angle_to_coords(math.radians(45), text_radius))

    circle = partial(Circle, cx=center, cy=center)

    return Svg(
        Defs(
            gradient_def,
            Mask(Rect(x=0, y=0, width=size, height=size, fill="white"), mask_path, id="dialMask"),
            Filter(FeDropShadow(dx="4", dy="4", stdDeviation="3", flood_opacity="0.3"), id="dropShadow",),
            text_path, m_text_path,
            Mask(                
                Rect(fill="black", **rect_attrs),
                Text(
                    f"{z_triangle} {abs(z):.4f}",
                    x=size/2, y=rect_y + rect_height/2,
                    font_family="Arial, sans-serif", font_size=size/18, font_weight="bold",
                    text_anchor="middle", dominant_baseline="central", fill="white"
                ),
                id="z-text-mask",
            )            
        ),
        circle(r=outer_radius, fill="url(#dialGradient)", mask="url(#dialMask)"),
        G(
            circle(r=inner_radius, fill="white", stroke="none"),
            radio_dial_arc, knob_path, *ticks, std_area, *std_dev_labels,                        
            m_value_text, m_indicator, data_text, z_background, z_cutout,
            filter="url(#dropShadow)"
        ),        
       
        Text(
            TextPath(
                Tspan(f"{gradient:.4f}", dy="0.4em"),
                Tspan(emoji, font_size=size/10, dy="0em"),
                href="#textPath", startOffset="50%",
            ),
            font_family="Arial, sans-serif", font_size=size/10, text_anchor="middle", fill="black"
        ),
        width=size, height=size,
        id="parameter-dial"
    )

@rt('/parameter_dial')
def get():
    random_data = random.uniform(-1, 1)
    random_m = random.uniform(-0.02, 0.02)
    random_v = random.uniform(0.003, 0.03)   
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
                ) for value, label in [("-0.8000", "Low (<0)"), ("0.0000", "Mid (0)"), (".0100", "High (>0)")]], 
                cls="radio-group"
            ),
            cls="mt-1 mb-4"
        ),
        create_parameter_dial(data=random_data, gradient=0.0000, m=random_m, v=random_v),
        cls="p-6 rounded-lg shadow-md mx-auto min-h-screen bg-[#e0e8d8] text-black"
    )

@rt("/update-dial")
async def update_dial(request: Request):
    form = await request.form()
    gradient = float(form.get("gradient", 0))
    random_data = random.uniform(-1, 1)
    random_m = random.uniform(-0.5, 0.5)
    random_v = random.uniform(0.5, 1.5)
    return create_parameter_dial(data=random_data, gradient=gradient)

# ------------------------------------------------------------------------------------------
# ----------------------------------------Gaussian------------------------------------------
# ------------------------------------------------------------------------------------------

@rt('/hovering_gaussian')
def get():    
    mean, std_dev = generate_gaussian_params()
    size = 400
    return Div(
        H1("Hovering Gaussian", cls="text-center mb-4"),
        P(f"Mean: {mean:.4f}, StdDev: {std_dev:.4f}", id="gaussian-params", cls="text-center mb-4"),        
        create_hovering_gaussian_svg(mean, std_dev, size),
        id="gaussian-container",
        cls="container mx-auto px-4"
    )

def generate_gaussian_params():
    mean = random.uniform(-0.5, 0.5)
    std_dev = math.sqrt(random.uniform(0.1, 0.5))
    return mean, std_dev


def create_hovering_gaussian_svg(mean, std_dev, size=400):
    gaussian_size = size * 0.375  # 150 / 400
    center = size // 2
    radial_segments, circular_segments = 40, 60

    # Create radial grid points
    points = [
        [(i / (radial_segments - 1))**2 * math.cos(angle), 
         (i / (radial_segments - 1))**2 * math.sin(angle)]
        for i in range(radial_segments)
        for angle in (2 * math.pi * j / circular_segments for j in range(circular_segments))
    ]

    # Create lines for the grid
    lines = [
        Line(0, 0, 0, 0, stroke="black", stroke_width=1, id=f"line_{i}")
        for i in range(radial_segments * circular_segments)
    ]

    # Create SVG content
    svg_content = Svg(
        G(*lines, id="gaussian"),
        width="100%", height="100%",
        viewBox=f"0 0 {size} {size}",
        preserveAspectRatio="xMidYMid meet",
        style="background-color: #f0f0f0; max-width: 100%; height: auto;"
    )
    
    gaussian_new_script = f"""
        const gaussianSize = {gaussian_size};
        const center = {center};
        const radialSegments = {radial_segments};
        const circularSegments = {circular_segments};
        window.mean = {mean};
        window.stdDev = {std_dev};

        const points = {json.dumps(points)};
        let is3D = false;
        let transitionProgress = 0;

        function gaussian(x, y, stdDev) {{
            const r = Math.sqrt(x*x + y*y);
            return Math.exp(-(r*r) / (2 * stdDev*stdDev));
        }}

        function rotatePoint(point, angleX, angleY) {{
            let [x, y, z] = point;
            const cosX = Math.cos(angleX), sinX = Math.sin(angleX);
            const cosY = Math.cos(angleY), sinY = Math.sin(angleY);
            
            const y1 = y * cosX - z * sinX;
            const z1 = y * sinX + z * cosX;
            
            return [
                x * cosY + z1 * sinY,
                y1,
                -x * sinY + z1 * cosY
            ];
        }}

        function updateGaussian() {{
            const time = performance.now() / 1000;
            const angleX = Math.sin(time * 0.3) * Math.PI / 12;
            const angleY = Math.sin(time * 0.4) * Math.PI / 8;
            const hoverOffset = Math.sin(time * 1.5) * 5;

            const rotatedPoints = points.map(p => {{
                const z = gaussian(p[0], p[1], window.stdDev) * transitionProgress;
                return rotatePoint([p[0], p[1], z], angleX * transitionProgress, angleY * transitionProgress);
            }});

            // Update grid lines
            for (let i = 0; i < radialSegments; i++) {{
                for (let j = 0; j < circularSegments; j++) {{
                    const lineIndex = i * circularSegments + j;
                    const line = document.getElementById(`line_${{lineIndex}}`);
                    if (line) {{
                        const start = rotatedPoints[lineIndex];
                        const endIndex = (j === circularSegments - 1) ? i * circularSegments : lineIndex + 1;
                        const end = rotatedPoints[endIndex];

                        line.setAttribute('x1', start[0] * gaussianSize + center);
                        line.setAttribute('y1', -start[2] * gaussianSize + center + hoverOffset * transitionProgress);
                        line.setAttribute('x2', end[0] * gaussianSize + center);
                        line.setAttribute('y2', -end[2] * gaussianSize + center + hoverOffset * transitionProgress);
                    }}
                }}
            }}

            if (is3D && transitionProgress < 1) {{
                transitionProgress += 0.02;
                if (transitionProgress > 1) transitionProgress = 1;
            }} else if (!is3D && transitionProgress > 0) {{
                transitionProgress -= 0.02;
                if (transitionProgress < 0) transitionProgress = 0;
            }}

            requestAnimationFrame(updateGaussian);
        }}

        function toggleView() {{
            is3D = !is3D;
        }}

        updateGaussian();

        document.getElementById('gaussian-container').addEventListener('click', toggleView);
    """

    return Div(
        svg_content,
        Script(gaussian_new_script),
        cls="w-full max-w-md mx-auto aspect-square"
    )

# Run the app
serve()
