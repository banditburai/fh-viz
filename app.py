import math
from fasthtml.common import *
from fasthtml.svg import *
from graphviz import Digraph
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import json
import asyncio
import time

tailwindLink = Link(rel="stylesheet", href="assets/output.css", type="text/css")
interactLink = Script(src="https://cdn.jsdelivr.net/npm/interactjs/dist/interact.min.js")
sselink = Script(src="https://unpkg.com/htmx-ext-sse@2.2.1/sse.js")
chartlink = Script(src="https://cdn.jsdelivr.net/npm/chart.js")
svgpanzoomlink = Script(src="https://cdn.jsdelivr.net/npm/svg-pan-zoom@3.6.1/dist/svg-pan-zoom.min.js")
app, rt = fast_app(
    pico=False,    
    hdrs=(tailwindLink, sselink, chartlink, svgpanzoomlink, interactLink)
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
        self.optimizer = AdamW(self.model.parameters(), lr=1e-1, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-4)

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
        self.train_split, self.val_split, self.test_split = gen_data_yinyang(random, n=100)
        
        # Reset optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=1e-1, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-4)

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
            # 1. Zero the gradients
            self.optimizer.zero_grad()
            # 2. Compute loss and backward pass
            loss = self.loss_fun(self.train_split)
            loss.backward()
            
            # 3. Optimizer step
            self.optimizer.step()

            self.step_count += 1
            self.train_losses.append(loss.data)

            # Evaluate validation loss every 10 steps
            if self.step_count % 10 == 0 and self.val_split:
                val_loss = self.loss_fun(self.val_split)
                self.val_losses.append(val_loss.data)
            
        return self.get_current_step()

    def get_current_step(self):
        param_state = {}
        for i, p in enumerate(self.model.parameters()):
            param_state[f'param_{i}'] = {
                'value': p.data if isinstance(p.data, float) else float(p.data),
                'grad': float(p.grad) if p.grad is not None else 0,
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

    def loss_fun(self, split):
        total_loss = Value(0.0)
        for x, y in split:
            logits = self.model(x)
            loss = cross_entropy(logits, y)
            total_loss = total_loss + loss
        mean_loss = total_loss * (1.0 / len(split))
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
    progressTracker: ({ step_count, train_loss, val_loss, param_state }) => {        
        if (step_count != null) {
            elements.stepText().textContent = `Step ${step_count}/100`;
        }
        if (train_loss != null) {
            const trainLossElement = elements.trainLossText();
            trainLossElement.textContent = `Train loss: ${train_loss}`;
            trainLossElement.dataset.value = train_loss;
        }
        if (val_loss != null) {
            const valLossElement = elements.valLossText();
            valLossElement.textContent = `Validation loss: ${val_loss}`;
            valLossElement.dataset.value = val_loss;
        }
        elements.chartContainer().classList.toggle('hidden', step_count === 0);
        if (step_count > 0) {
            updateLossChart(step_count, parseFloat(train_loss), parseFloat(val_loss));
        }
        if (param_state) {
            updateParameterDials(param_state);
        }
        document.dispatchEvent(new CustomEvent('progressUpdated', { detail: { step_count, train_loss, val_loss, param_state } }));
    }
};

let dialConfig;

function initializeDialConfig() {
    const configElement = document.getElementById('dial-config');
    if (configElement && configElement.value) {
        try {
            dialConfig = JSON.parse(configElement.value);            
        } catch (error) {
            console.error('Error parsing dial configuration:', error);
            console.log('Raw configuration value:', configElement.value);
        }
    } else {
        console.error('Dial configuration element not found or empty');
    }
}


const updateParameterDials = (paramState) => {    
    if (!dialConfig) {
        console.error('Dial configuration not initialized. Cannot update dials.');
        return;
    }

    const optimizerParams = {
        lr: parseFloat(document.getElementById('lr-select').value),
        beta1: parseFloat(document.getElementById('beta1-select').value),
        beta2: parseFloat(document.getElementById('beta2-select').value),
        eps: parseFloat(document.getElementById('eps-select').value),
        weight_decay: parseFloat(document.getElementById('weight-decay-select').value)
    };
    
    Object.entries(paramState).forEach(([key, param]) => {
        const dial = document.getElementById(key);        
        if (!dial) {
            console.warn(`Dial not found for ${key}`);
            return;
        }

        const dataText = dial.querySelector('[data-id="data-value"]');
        const gradientText = dial.querySelector('tspan[data-id="gradient-value"]');       
        const mBeanGroup = dial.querySelector('[data-id="m-bean-group"]');
        const mIndicator = dial.querySelector('[data-id="m-indicator"]');
        const knobGroup = dial.querySelector('g[data-rotation-angle]');
        const gradientBase = dial.querySelector('use[href="#gradient-base"]');
        const sqrtVText = dial.querySelector('[data-id="sqrt-v-value"]');

        if (!dataText || !gradientText || !mIndicator || !knobGroup || !gradientBase) {
            console.warn(`Some elements not found for ${key}`);
            return;
        }

        // Update data value
        if (param.value !== undefined) {
            dataText.textContent = param.value.toFixed(4);
        }

        // Update gradient value
        if (param.grad !== undefined) {
            gradientText.textContent = param.grad.toFixed(4);
        }
        
        // Update gradient
        const gradientKey = param.grad < 0 ? 'negative' : param.grad > 0 ? 'positive' : 'zeroed';
        gradientBase.setAttribute('fill', `url(#dial-gradient-${gradientKey})`);
        gradientBase.dataset.gradientKey = gradientKey;

        // Update rotation
        try {
            const percentage = normalizeGradient(param.grad, dialConfig.SCALE_MIN, dialConfig.SCALE_MAX, dialConfig.LINTHRESH);
            if (isNaN(percentage)) {
                console.error(`Normalized gradient is NaN for ${key}. Gradient: ${param.grad}`);
                return;
            }
            const rotationAngle = 180 + percentage * 1.8;
            if (isNaN(rotationAngle)) {
                console.error(`Rotation angle is NaN for ${key}. Percentage: ${percentage}`);
                return;
            }
            knobGroup.setAttribute('transform', `rotate(${rotationAngle} ${dialConfig.CENTER} ${dialConfig.CENTER})`);
            knobGroup.dataset.rotationAngle = rotationAngle;
        } catch (error) {
            console.error(`Error updating rotation for ${key}:`, error);
        }

        const m_lookahead = optimizerParams.beta1 * param.m + (1 - optimizerParams.beta1) * param.grad;
        const v_lookahead = optimizerParams.beta2 * param.v + (1 - optimizerParams.beta2) * (param.grad ** 2);
        const sqrt_v_lookahead = Math.sqrt(v_lookahead + optimizerParams.eps);
        const z = -(optimizerParams.lr * m_lookahead) / sqrt_v_lookahead;  


        // Update m-related elements
        if (param.m !== undefined && param.grad !== undefined) {
            const m_angle = pos_to_angle(param.m, dialConfig);
            updateArrow(key, param.grad, m_angle, optimizerParams);            
            const outer_m_radius = dialConfig.INNER_RADIUS * 0.70;

            // Update m indicator
            const mIndicator = dial.querySelector('[data-id="m-indicator"]');
            if (mIndicator) {
                const [x2, y2] = angleToCoords(m_angle, outer_m_radius * 1.15, dialConfig.CENTER);
                mIndicator.setAttribute('x1', dialConfig.CENTER);
                mIndicator.setAttribute('y1', dialConfig.CENTER);
                mIndicator.setAttribute('x2', x2);
                mIndicator.setAttribute('y2', y2);
            }

            const buffer_angle = 12 * Math.PI / 180;
            const text_arc_length = 40 * Math.PI / 180;
            let mBeanGroup = dial.querySelector('[data-id="m-bean-group"]');
            if (param.m !== 0) {
                if (!mBeanGroup) {
                    mBeanGroup = createMBeanGroup(key, param.m, dialConfig);
                    dial.appendChild(mBeanGroup);
                }                                
                const adjusted_angle = param.m >= 0 ? m_angle - buffer_angle - text_arc_length : m_angle + buffer_angle;
                rotation_degrees = adjusted_angle * 180 / Math.PI;
                
                mBeanGroup.setAttribute('transform', `rotate(${rotation_degrees} ${dialConfig.CENTER} ${dialConfig.CENTER})`);

                // Update the text content
                const textPath = mBeanGroup.querySelector('textPath');
                if (textPath) {
                    textPath.textContent = param.m.toFixed(4);
                }
            } else if (mBeanGroup) {
                mBeanGroup.remove();
            }            
        }        
        updateZShape(key, z);

        if (sqrtVText) {
            sqrtVText.textContent = sqrt_v_lookahead.toFixed(4);
        }

    });
};

function createArrowPath(centerX, centerY, gradient, maxGradient, mAngle, innerRadius, size) {
    const arcRadius = innerRadius * 0.76;
    const start = {
        x: centerX + arcRadius * Math.cos(mAngle),
        y: centerY + arcRadius * Math.sin(mAngle)
    };

    const minLength = Math.PI / 36; // Minimum 5 degrees for visibility
    const maxLength = Math.PI / 2; // Maximum 90 degrees
    let gradientLength = Math.max(Math.min(Math.abs(gradient) / maxGradient * maxLength, maxLength), minLength);
    gradientLength *= Math.sign(gradient);

    const endAngle = mAngle + gradientLength;
    const end = {
        x: centerX + arcRadius * Math.cos(endAngle),
        y: centerY + arcRadius * Math.sin(endAngle)
    };

    if (gradient === 0) {
        return null;
    }

    const baseArrowLength = size / 30;
    const baseArrowWidth = size / 90;
    const scaleFactor = 0.5 + 0.5 * Math.min(Math.abs(gradient) / maxGradient, 1);
    const arrowLength = baseArrowLength * scaleFactor;
    const arrowWidth = baseArrowWidth * scaleFactor;

    const angle = Math.atan2(end.y - start.y, end.x - start.x);
    const cosAngle = Math.cos(angle);
    const sinAngle = Math.sin(angle);

    const arrowhead = [
        end,
        {
            x: end.x - arrowLength * cosAngle + arrowWidth * sinAngle,
            y: end.y - arrowLength * sinAngle - arrowWidth * cosAngle
        },
        {
            x: end.x - arrowLength * cosAngle - arrowWidth * sinAngle,
            y: end.y - arrowLength * sinAngle + arrowWidth * cosAngle
        }
    ];

    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("stroke", "blue");
    path.setAttribute("stroke-width", "2");
    path.setAttribute("fill", "blue");

    const d = [
        `M ${start.x} ${start.y}`,
        `A ${arcRadius} ${arcRadius} 0 0 ${gradient > 0 ? 1 : 0} ${end.x} ${end.y}`,
        `M ${arrowhead[0].x} ${arrowhead[0].y}`,
        `L ${arrowhead[1].x} ${arrowhead[1].y}`,
        `L ${arrowhead[2].x} ${arrowhead[2].y}`,
        "Z"
    ].join(" ");

    path.setAttribute("d", d);
    return path;
}

function updateArrow(key, gradient, mAngle, optimizerState) {
    const dial = document.getElementById(key);
    if (!dial) return;

    const config = JSON.parse(document.getElementById('dial-config').value);
    const { CENTER, INNER_RADIUS, OUTER_RADIUS, SIZE } = config;

    let arrowGroup = dial.querySelector('[data-id="arrow-group"]');
    if (!arrowGroup) {
        arrowGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
        arrowGroup.setAttribute("data-id", "arrow-group");
        dial.appendChild(arrowGroup);
    }

    // Clear existing arrow
    while (arrowGroup.firstChild) {
        arrowGroup.removeChild(arrowGroup.firstChild);
    }

    const maxGradient = 5.0;
    const arrowPath = createArrowPath(CENTER, CENTER, gradient, maxGradient, mAngle, INNER_RADIUS, SIZE);

    if (arrowPath) {
        arrowGroup.appendChild(arrowPath);
    }
}


function updateZShape(paramName, z) {
    const zShapeGroup = document.querySelector(`[data-id="${paramName}-z-shape"]`);
    if (zShapeGroup) {
        const zShape = zShapeGroup.querySelector('use');
        const newShapeId = z >= 0 ? "base-z-shape-positive" : "base-z-shape-negative";
        zShape.setAttribute('href', `#${newShapeId}`);
        
        const text = zShapeGroup.querySelector('[data-id="z-value"]');
        if (text) {
            const triangle = z >= 0 ? "▲" : "▼";
            text.textContent = `${triangle} ${Math.abs(z).toFixed(4)}`;
        } else {
            console.warn(`Z-value text element not found for ${paramName}`);
        }
    } else {
        console.warn(`Z-shape group not found for ${paramName}`);
    }
}

function createMBeanGroup(paramName, m, config) {
    const mBeanGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
    mBeanGroup.setAttribute("data-id", "m-bean-group");

    const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
    use.setAttributeNS("http://www.w3.org/1999/xlink", "xlink:href", "#base-m-bean");
    mBeanGroup.appendChild(use);

    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("font-family", "Arial, sans-serif");
    text.setAttribute("font-size", config.SIZE / 30);
    text.setAttribute("fill", "white");
    text.setAttribute("text-anchor", "middle");
    text.setAttribute("dominant-baseline", "central");

    const textPath = document.createElementNS("http://www.w3.org/2000/svg", "textPath");
    textPath.setAttributeNS("http://www.w3.org/1999/xlink", "xlink:href", "#base-m-text-path");
    textPath.setAttribute("startOffset", "50%");
    textPath.textContent = m.toFixed(4);

    text.appendChild(textPath);
    mBeanGroup.appendChild(text);

    return mBeanGroup;
}

function symlog_scale(x, linthresh) {
    return Math.sign(x) * Math.log1p(Math.abs(x) / linthresh);
}

function pos_to_angle(pos, config) {
    const log_scale_min = symlog_scale(config.SCALE_MIN, config.LINTHRESH);
    const log_scale_max = symlog_scale(config.SCALE_MAX, config.LINTHRESH);
    return Math.PI + (symlog_scale(pos, config.LINTHRESH) - log_scale_min) / (log_scale_max - log_scale_min) * Math.PI;
}

function angleToCoords(angle, radius, center) {
    return [
        center + radius * Math.cos(angle),
        center + radius * Math.sin(angle)
    ];
}

function normalizeGradient(gradient, scaleMin, scaleMax, linthresh) {
    const symlogScale = x => Math.sign(x) * Math.log1p(Math.abs(x) / linthresh);
    const logGradient = symlogScale(gradient);
    const logMin = symlogScale(scaleMin);
    const logMax = symlogScale(scaleMax);
    return (logGradient - logMin) / (logMax - logMin) * 100;
}

const updateSSEConnection = () => {
    if (state.eventSource) {
        console.log('Closing existing SSE connection');
        state.eventSource.close();
    }
    
    if (state.reconnectAttempts >= config.maxReconnectAttempts) {
        console.error('Max reconnection attempts reached. Please refresh the page.');
        return;
    }

    console.log('Opening new SSE connection');
    state.eventSource = new EventSource(`/train_stream?train=${state.isTraining}`);
    
    state.eventSource.addEventListener('open', () => {
        console.log('SSE connection opened successfully');
        state.reconnectAttempts = 0;
    });

    state.eventSource.addEventListener('step', (e) => {        
        updateUI.progressTracker(JSON.parse(e.data));
    });

    state.eventSource.addEventListener('heartbeat', () => {
        console.log('Heartbeat received');
    });

    state.eventSource.addEventListener('error', (e) => {
        console.error('SSE connection error:', e);
        state.eventSource.close();
        state.reconnectAttempts++;
        console.log(`Attempting to reconnect (attempt ${state.reconnectAttempts})`);
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
    initializeDialConfig(); 
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

@rt('/')
def get():
    dial_config = DialConfig()
    
    dial_config_json = json.dumps({
        "CENTER": dial_config.CENTER,
        "INNER_RADIUS": dial_config.INNER_RADIUS,
        "OUTER_RADIUS": dial_config.OUTER_RADIUS,
        "SCALE_MIN": dial_config.SCALE_MIN,
        "SCALE_MAX": dial_config.SCALE_MAX,
        "LINTHRESH": dial_config.LINTHRESH,
        "RING_WIDTH": dial_config.RING_WIDTH,
        "SIZE": dial_config.SIZE
    })

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
                        Input(type="hidden", id="dial-config", value=dial_config_json),
                        create_main_parameter_dials(visualizer, dial_config=dial_config),
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

class DialConfig(BaseModel):
    SIZE: int = 200
    SCALE_FACTOR: float = 0.4
    SPACING: int = 10
    RING_WIDTH: float = SIZE / 8
    OUTER_RADIUS: float = SIZE / 2 - 5
    INNER_RADIUS: float = OUTER_RADIUS - RING_WIDTH
    CENTER: float = SIZE / 2
    SCALE_MIN: float = -1.0
    SCALE_MAX: float = 1.0
    LINTHRESH: float = 0.01

class ParameterState(BaseModel):
    name: str
    data: float
    gradient: float
    m: float
    v: float

class OptimizerState(BaseModel):
    t: int
    lr: float
    beta1: float
    beta2: float
    eps: float
    weight_decay: float

class TrainingUpdate(BaseModel):
    step: int
    parameters: Dict[str, ParameterState]
    optimizer: OptimizerState

def calculate_grid_layout(total_params: int, dial_config: DialConfig, container_width: int, container_height: int) -> Dict[str, int]:
    scaled_size = dial_config.SIZE * dial_config.SCALE_FACTOR
    max_columns = max(1, (container_width + dial_config.SPACING) // (scaled_size + dial_config.SPACING))
    rows = math.ceil(total_params / max_columns)
    columns = min(max_columns, total_params)
    
    grid_width = columns * (scaled_size + dial_config.SPACING) - dial_config.SPACING
    grid_height = rows * (scaled_size + dial_config.SPACING) - dial_config.SPACING
    
    start_x = (container_width - grid_width) / 2
    start_y = (container_height - grid_height) / 2
    
    return {
        "columns": columns,
        "rows": rows,
        "grid_width": grid_width,
        "grid_height": grid_height,
        "start_x": start_x,
        "start_y": start_y,
        "svg_width": max(container_width, grid_width + 2 * start_x),
        "svg_height": max(container_height, grid_height + 2 * start_y),
    }

def create_main_parameter_dials(visualizer, container_width=800, container_height=400, dial_config=None):
    if not visualizer.model:
        return Div("No model available", id="parameter-dials")    
    
    current_step = visualizer.get_current_step()
    param_states = current_step['param_state']

    total_params = sum(len(neuron.w) + 1 for layer in visualizer.model.layers for neuron in layer.neurons)
    layout = calculate_grid_layout(total_params, dial_config, container_width, container_height)

    optimizer_state = visualizer.get_optimizer_state()
    
    base_dial = create_base_dial(dial_config)
    base_knob = create_base_knob_path(dial_config)    
    base_m_bean = create_base_m_bean(dial_config)
    z_shape_positive, z_shape_negative = create_z_shapes(dial_config)
    common_defs = Defs(
        G(base_dial, id="base-dial"),        
        Mask(
        Rect(x=0, y=0, width=dial_config.SIZE, height=dial_config.SIZE, fill="white"),
        create_outer_dial_mask_path(dial_config),
        id="dialMask"
    ),
        Mask(Rect(x=0, y=0, width=dial_config.SIZE, height=dial_config.SIZE, fill="white"),
             Rect(x=0, y=dial_config.CENTER, width=dial_config.SIZE, height=dial_config.CENTER, fill="black"),
             id="top-half-mask"
                 ),
        Filter(
            FeDropShadow(dx="4", dy="4", stdDeviation="3", flood_opacity="0.3"),
            id="dropShadow"
        ),                
        create_outer_dial_gradient_text_path((dial_config.OUTER_RADIUS + dial_config.INNER_RADIUS) / 2, dial_config.CENTER),
        *setup_dial_gradients(),    
        Circle(dial_config.OUTER_RADIUS, dial_config.CENTER, dial_config.CENTER, mask="url(#dialMask)", id="gradient-base"),        
        G(            
            Circle(dial_config.INNER_RADIUS, dial_config.CENTER, dial_config.CENTER, fill="white", stroke="none"),
            base_knob,            
            id="knob-circle-group"
        ), 
       G(base_m_bean, id="base-m-bean"),
       z_shape_positive, z_shape_negative,
       create_base_sqrt_v_definitions(dial_config),
       create_sqrt_v_gradients(),
       create_sqrt_v_shadow_gradient()       
    )

    dial_elements = [
        create_positioned_dial(i, param_states[f'param_{i}'], layout, dial_config, optimizer_state)
        for i in range(total_params)
    ]

    return Svg(
        common_defs,      
        *dial_elements,    
        width="100%", height="100%", 
        viewBox=f"0 0 {layout['svg_width']} {layout['svg_height']}",
        preserveAspectRatio="xMidYMid meet",
        id="parameter-dials-svg",
        style="min-height: 400px;"
    )    

def create_positioned_dial(i: int, param_state: Dict, layout: Dict[str, int], dial_config: DialConfig, optimizer_state: Dict):
    scaled_size = dial_config.SIZE * dial_config.SCALE_FACTOR
    cx = layout['start_x'] + (i % layout['columns']) * (scaled_size + dial_config.SPACING)
    cy = layout['start_y'] + (i // layout['columns']) * (scaled_size + dial_config.SPACING)
    
    dial = create_parameter_dial(f"param_{i}", param_state['value'], param_state['grad'], param_state['m'], param_state['v'], optimizer_state)
    return G(dial, transform=f"translate({cx},{cy}) scale({dial_config.SCALE_FACTOR})")

def create_base_dial(config: DialConfig):
    CENTER = config.CENTER
    outer_m_radius = config.INNER_RADIUS * 0.70
    inner_v_radius = outer_m_radius * 0.6

    base_dial = G(id="base-dial")

    m_circle = Circle(
        outer_m_radius,
        CENTER, CENTER,
        fill="none",
        stroke="#888",
        stroke_width=1,
        mask="url(#top-half-mask)"
    )
    base_dial(*[m_circle])

    v_circle = Circle(
        inner_v_radius,
        CENTER, CENTER,
        fill="none",
        stroke="#888",
        stroke_width=1,
        mask="url(#top-half-mask)"
    )
    base_dial(*[v_circle])
    
    v_ticks = create_v_ticks(inner_v_radius, CENTER)
    base_dial(*[v_ticks])
    
    m_labels_and_ticks = create_m_labels_and_ticks(config)
    base_dial(*[m_labels_and_ticks])
    
    return base_dial

def create_base_m_bean(config):
    outer_m_radius = config.INNER_RADIUS * 0.70
    inner_v_radius = outer_m_radius * 0.6
    m_text_radius = (outer_m_radius + inner_v_radius) * 0.5

    outer_bean_radius = m_text_radius * 1.15
    inner_bean_radius = m_text_radius * 0.85
    text_radius = (outer_bean_radius + inner_bean_radius) / 2
        
    start_angle = 0
    end_angle = math.radians(40)
    center = config.CENTER
    edge_radius = (outer_bean_radius - inner_bean_radius) / 2

    bean_path = (Path(fill="#372572", opacity="0.8", id="base-m-bean")
            .M(center + outer_bean_radius * math.cos(start_angle), 
               center + outer_bean_radius * math.sin(start_angle))
            .A(outer_bean_radius, outer_bean_radius, 0, 0, 1, 
               center + outer_bean_radius * math.cos(end_angle), 
               center + outer_bean_radius * math.sin(end_angle))
            .A(edge_radius, edge_radius, 0, 0, 1,
               center + inner_bean_radius * math.cos(end_angle),
               center + inner_bean_radius * math.sin(end_angle))
            .A(inner_bean_radius, inner_bean_radius, 0, 0, 0,
               center + inner_bean_radius * math.cos(start_angle),
               center + inner_bean_radius * math.sin(start_angle))
            .A(edge_radius, edge_radius, 0, 0, 1,
               center + outer_bean_radius * math.cos(start_angle),
               center + outer_bean_radius * math.sin(start_angle))
            .Z())
         
    text_path = (Path(fill="none", id="base-m-text-path")
        .M(center + text_radius * math.cos(start_angle), 
           center + text_radius * math.sin(start_angle))
        .A(text_radius, text_radius, 0, 0, 1, 
           center + text_radius * math.cos(end_angle), 
           center + text_radius * math.sin(end_angle))
    )


    return G(bean_path, text_path)

def create_z_shapes(config: DialConfig):
    SIZE = config.SIZE
    INNER_RADIUS = config.INNER_RADIUS
    rect_width, rect_height = SIZE * 0.4, SIZE * 0.07
    rect_x = SIZE / 2 - rect_width / 2
    rect_y = SIZE / 2 + INNER_RADIUS / 2.4

    def create_shape(color):
        return G(
            Rect(
                x=rect_x, y=rect_y,
                width=rect_width, height=rect_height,
                rx=rect_height / 2,
                fill=color, opacity=0.8
            )
        )

    return (
        G(create_shape("#4CAF50"), id="base-z-shape-positive"),
        G(create_shape("#F44336"), id="base-z-shape-negative")
    )

def create_outer_dial_gradient_text_path(text_radius, CENTER):
    start = angle_to_coords(math.radians(135), text_radius, CENTER)
    end = angle_to_coords(math.radians(45), text_radius, CENTER)
    
    return (Path(id="gradientTextPath", stroke="purple", stroke_width=2, fill="none")
            .M(*start)
            .A(text_radius, text_radius, 0, 0, 0, *end))

def create_base_knob_path(config: DialConfig):
    knob_width, knob_length = config.RING_WIDTH, config.RING_WIDTH * 1.1
    knob_angle = 0 

    knob_x, knob_y = angle_to_coords(knob_angle, config.OUTER_RADIUS, config.CENTER)
    knob_end_x, knob_end_y = angle_to_coords(knob_angle, config.OUTER_RADIUS - knob_length, config.CENTER)

    return (Path(fill="white")
            .M(knob_x, knob_y)
            .L(knob_end_x + knob_width/2 * math.sin(knob_angle), 
               knob_end_y - knob_width/2 * math.cos(knob_angle))
            .L(knob_end_x - knob_width/2 * math.sin(knob_angle), 
               knob_end_y + knob_width/2 * math.cos(knob_angle))
            .Z())

def calculate_updates(param: ParameterState, optimizer: OptimizerState):
    m_lookahead = optimizer.beta1 * param.m + (1 - optimizer.beta1) * param.gradient
    v_lookahead = optimizer.beta2 * param.v + (1 - optimizer.beta2) * (param.gradient ** 2)
    sqrt_v_lookahead = math.sqrt(v_lookahead + optimizer.eps)
    update_viz = -m_lookahead / sqrt_v_lookahead
    
    if optimizer.t > 0:
        m_hat = m_lookahead / (1 - optimizer.beta1 ** optimizer.t)
        v_hat = v_lookahead / (1 - optimizer.beta2 ** optimizer.t)
        update_actual = optimizer.lr * (m_hat / (math.sqrt(v_hat) + optimizer.eps) + optimizer.weight_decay * param.data)
    else:
        update_actual = 0
    
    return m_lookahead, v_lookahead, update_viz, update_actual

def create_m_labels_and_ticks(config: DialConfig):
    outer_m_radius = config.INNER_RADIUS * 0.70
    m_ticks = []
    m_labels = []
    label_distance = 0.7
    m_values = [-1.00, -0.20, -0.02, 0.00, 0.02, 0.20, 1.00]

    for i, value in enumerate(m_values):
        angle = pos_to_angle(value, config)
        tick_start = angle_to_coords(angle, outer_m_radius, config.CENTER)
        tick_end = angle_to_coords(angle, outer_m_radius + 5, config.CENTER)
        m_ticks.append(Line(x1=tick_start[0], y1=tick_start[1], x2=tick_end[0], y2=tick_end[1], stroke="#888", stroke_width=.8))
        
        text_width = len(f"{value:.2f}") * config.SIZE / 40
        label_radius = outer_m_radius + label_distance + text_width / 2
        label_pos = angle_to_coords(angle, label_radius, config.CENTER)
        
        rotation = (angle - math.pi) * 180 / math.pi
        if rotation > 90 or rotation < -90:
            rotation += 180
        
        m_labels.append(
            Text(
                f"{value:.2f}",
                x=label_pos[0], y=label_pos[1],
                font_size=config.SIZE/40,
                fill="#888",
                text_anchor="middle",
                dominant_baseline="central",
                transform=f"rotate({rotation} {label_pos[0]} {label_pos[1]})"
            )
        )
    
    return G(id="m-labels-and-ticks", *m_ticks, *m_labels)

def create_v_ticks(inner_v_radius, CENTER, num_ticks=20):
    return G(id="v-ticks", *[
        Line(
            *sum((angle_to_coords(angle, r, CENTER) for r in (inner_v_radius, inner_v_radius - 5)), ()),
            stroke="#888", stroke_width=1
        )
        for i in range(num_ticks + 1)
        for angle in [math.pi + (i / num_ticks) * math.pi]
    ])

def create_outer_dial_mask_path(config: DialConfig):
    START_ANGLE, END_ANGLE = math.radians(135), math.radians(45)
    center = config.SIZE / 2
    radius = config.OUTER_RADIUS
    
    start = angle_to_coords(START_ANGLE, radius, center)
    end = angle_to_coords(END_ANGLE, radius, center)
    
    return Path(fill="black").M(center, center).L(*start).A(radius, radius, 0, 1, 0, *end).Z()


def create_parameter_dial(param_name: str, data: float, gradient: float, m: float, v: float, 
                          optimizer_state: OptimizerState, config: DialConfig = DialConfig()):
    SIZE = config.SIZE
    CENTER = config.CENTER
    OUTER_RADIUS, INNER_RADIUS = config.OUTER_RADIUS, config.INNER_RADIUS
    RING_WIDTH = config.RING_WIDTH
    SCALE_MIN, SCALE_MAX = config.SCALE_MIN, config.SCALE_MAX
    LINTHRESH = config.LINTHRESH        
         
    gradient_key = 'negative' if gradient < 0 else 'positive' if gradient > 0 else 'zeroed'
    gradient_id = f"dial-gradient-{gradient_key}"
    percentage = normalize_gradient(gradient, scale_min=SCALE_MIN, scale_max=SCALE_MAX, linthresh=LINTHRESH)
    knob_rotation_angle = 180 + percentage * 1.8   
        
    outer_m_radius = INNER_RADIUS * 0.70 
    inner_v_radius = outer_m_radius * 0.6 
    m_text_radius = (outer_m_radius + inner_v_radius) * .5
                
    m_angle = pos_to_angle(m, config)    
    
    m_bean_group = G(
        Use(href="#base-m-bean"),        
        Text(
            TextPath(
            f"{m:.4f}",
            href="#base-m-text-path",
            startOffset="50%"
            ),
            font_family="Arial, sans-serif",
            font_size=config.SIZE/30,
            font_weight="bold",
            fill="white",
            text_anchor="middle",
            dominant_baseline="central"
        ),
        id=f"{param_name}-m-bean-group",        
        data_id="m-bean-group"
    ) if m != 0 else None

    buffer_angle = math.radians(12) 
    text_arc_length = math.radians(40)  

    if m >= 0:
        adjusted_angle = m_angle - buffer_angle - text_arc_length
    else:
        adjusted_angle = m_angle + buffer_angle

    rotation_degrees = math.degrees(adjusted_angle)

    rotated_m_bean_group = G(
        m_bean_group, 
        transform=f"rotate({rotation_degrees} {config.CENTER} {config.CENTER})"
    ) if m_bean_group else None

    m_indicator = Line(
        x1=CENTER,
        y1=CENTER,
        x2=angle_to_coords(m_angle, outer_m_radius * 1.15, CENTER)[0],
        y2=angle_to_coords(m_angle, outer_m_radius * 1.15, CENTER)[1],
        stroke="#372572", stroke_width=2, data_id="m-indicator"
    )   
        
    arrow_length = abs(gradient) * (1 - optimizer_state['beta1']) * (OUTER_RADIUS - INNER_RADIUS)        
    max_arrow_length = (OUTER_RADIUS - INNER_RADIUS) * 0.5
    arrow_length = min(arrow_length, max_arrow_length)
            
    max_gradient = 5.0
    blue_arrow = create_arrow_path(CENTER, CENTER, gradient, max_gradient, m_angle, INNER_RADIUS, SIZE)
    
    sqrt_v_width = SIZE * 0.20 
    sqrt_v_height = SIZE * 0.10
    sqrt_v_x = CENTER - sqrt_v_width/2
    sqrt_v_y = CENTER - SIZE/30 
    cylinder_height = sqrt_v_height * 0.3

    sqrt_v_shape = G(                
        Use(href="#base-sqrt-v-cylinder"),        
        Text(
            f"{math.sqrt(v):.4f}",
            x=CENTER, y=sqrt_v_y + cylinder_height * 0.8,
            font_family="Arial, sans-serif",
            font_size=SIZE/25,
            font_weight="bold",
            text_anchor="middle",
            dominant_baseline="central",
            fill="#FFFFFF",
            data_id="sqrt-v-value"
        ),
        id=f"{param_name}-sqrt-v-visualization"
    )
    data_text = Text(
        f"{data:.4f}", x=SIZE/2, y=SIZE/2 + INNER_RADIUS/4,
        font_family="Arial, sans-serif", font_size=SIZE/8, font_weight="bold",
        text_anchor="middle", dominant_baseline="central", fill="black", data_id="data-value"
    )

    z_shape_group = G(
        Use(href="#base-z-shape-positive"),
        Text(
            "▲ 0.0000",
            x=config.SIZE/2, y=config.SIZE/2 + config.INNER_RADIUS/2.4 + config.SIZE*0.07/2,
            font_family="Arial, sans-serif", font_size=config.SIZE/18, font_weight="bold",
            text_anchor="middle", dominant_baseline="central", fill="white",
            data_id="z-value"
        ),
        data_id=f"{param_name}-z-shape"
    )
    return Svg(        
        Use(href="#gradient-base", fill=f"url(#{gradient_id})", data_gradient_key=gradient_key),
        G(
            Use(href="#knob-circle-group"),
            transform=f"rotate({knob_rotation_angle} {config.CENTER} {config.CENTER})",
            filter="url(#dropShadow)",
            data_rotation_angle=knob_rotation_angle
        ),
        G(            
            Use(href="#base-dial"),          
            blue_arrow,
            data_text, z_shape_group,                                                     
            m_indicator,
            rotated_m_bean_group,
            sqrt_v_shape, 
        ),        
       
        Text(
            TextPath(
                Tspan(f"{gradient:.4f}", dy="0.4em", data_id="gradient-value"),                
                href="#gradientTextPath", startOffset="50%",
            ),
            font_family="Arial, sans-serif", font_size=SIZE/10, text_anchor="middle", fill="black"
        ),
        id=param_name,
        width=SIZE, height=SIZE,
        viewBox=f"0 0 {SIZE} {SIZE}",                
    )     

def setup_dial_gradients():
        gradient_defs = {
            'negative': ("#240b36", "#4a1042", "#711845", "#981f3c", "#b72435", "#c31432"),
            'zeroed': ("#fdc830", "#fdc130", "#fcb130", "#fba130", "#f99130", "#f37335"),
            'positive': ("#11998e", "#1eac8e", "#2aba8e", "#35c78d", "#37d18b", "#38db89", "#38ef7d")
        }

        return [
            LinearGradient(
                *[Stop(offset=f"{i/(len(colors)-1)*100}%", 
                    style=f"stop-color:{color};stop-opacity:1")
                for i, color in enumerate(colors)],
                id=f"dial-gradient-{key}", x1="0%", y1="0%", x2="100%", y2="100%",
            ) for key, colors in gradient_defs.items()
        ]

def create_base_sqrt_v_definitions(config: DialConfig):
    sqrt_v_width = config.SIZE * 0.20 
    sqrt_v_height = config.SIZE * 0.10
    cylinder_height = sqrt_v_height * 0.3
    sqrt_v_x = config.CENTER - sqrt_v_width/2
    sqrt_v_y = config.CENTER - config.SIZE/30 

    base_cylinder = G(
        # Top ellipse
        Ellipse(
            sqrt_v_width/2, sqrt_v_height/2,
            cx=config.CENTER, cy=sqrt_v_y,
            fill="url(#base-sqrt-v-top-gradient)",
            stroke="none",
        ),
        # Side
        Path(fill="url(#base-sqrt-v-side-gradient)", stroke="none")
            .M(sqrt_v_x, sqrt_v_y)
            .Q(config.CENTER, sqrt_v_y + cylinder_height * 1.2, sqrt_v_x + sqrt_v_width, sqrt_v_y)
            .L(sqrt_v_x + sqrt_v_width, sqrt_v_y + cylinder_height)
            .Q(config.CENTER, sqrt_v_y + cylinder_height * 0.8, sqrt_v_x, sqrt_v_y + cylinder_height)
            .Z(),
        # Bottom ellipse
        Ellipse(
            sqrt_v_width/2, sqrt_v_height/4,
            cx=config.CENTER, cy=sqrt_v_y + cylinder_height,
            fill="url(#base-sqrt-v-bottom-gradient)",
            stroke="none",
        ),
         # Shadow
        Rect(
            x=sqrt_v_x, 
            y=sqrt_v_y + cylinder_height,
            width=sqrt_v_width,
            height=config.SIZE * 0.04, 
            fill="url(#base-sqrt-v-shadow-gradient)",
            stroke="none",
        ),
        id="base-sqrt-v-cylinder"
    )

    return base_cylinder

def create_sqrt_v_gradients():
    return G(
        RadialGradient(
            Stop(offset="0%", stop_color="white"),
            Stop(offset="70%", stop_color="#4a90e2"),
            Stop(offset="100%", stop_color="#2171cd"),
            id="base-sqrt-v-top-gradient",
            cx="50%", cy="50%", r="50%", fx="25%", fy="25%"
        ),
        LinearGradient(
            Stop(offset="0%", stop_color="#4a90e2"),
            Stop(offset="100%", stop_color="#2171cd"),
            id="base-sqrt-v-side-gradient",
            x1="0%", y1="0%", x2="0%", y2="100%"
        ),
        RadialGradient(
            Stop(offset="0%", stop_color="#2171cd"),
            Stop(offset="100%", stop_color="#4a90e2"),
            id="base-sqrt-v-bottom-gradient",
            cx="50%", cy="50%", r="50%", fx="25%", fy="25%"
        ),
    )

def create_sqrt_v_shadow_gradient():
    return LinearGradient(
        Stop(offset="0%", stop_color="rgba(0,0,0,0.3)"),
        Stop(offset="100%", stop_color="rgba(0,0,0,0)"),
        id="base-sqrt-v-shadow-gradient",
        x1="0%", y1="0%", x2="0%", y2="100%"
    )

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

def pos_to_angle(pos, config):
    log_scale_min = symlog_scale(config.SCALE_MIN, config.LINTHRESH)
    log_scale_max = symlog_scale(config.SCALE_MAX, config.LINTHRESH)
    return math.pi + (symlog_scale(pos, config.LINTHRESH) - log_scale_min) / (log_scale_max - log_scale_min) * math.pi

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

def create_base_m_bean_definitions(config: DialConfig):
    radii = (config.INNER_RADIUS * 0.70 * factor for factor in (1.15, 0.85))
    bean_shape = create_bean_shape(config.CENTER, config.CENTER, *radii, 0, math.radians(40))
    bean_shape.fill = "#372572"
    bean_shape.id = "base-m-bean"
    return bean_shape

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


@rt("/resize")
def get():
    return Div(
        Style("""
            .grid-container {
                display: grid;
                grid-template-columns: 1fr 10px 1fr;
                grid-template-rows: 1fr 10px 1fr;
                gap: 0;
                height: 100vh;
                width: 100vw;
            }
            .resizable {
                background-color: #f0f0f0;
                overflow: hidden;
            }
            .gutter {
                background-color: #ddd;
                transition: background-color 0.3s;
            }
            .gutter:hover {
                background-color: #4299e1;
            }
            .gutter-horizontal {
                cursor: col-resize;
            }
            .gutter-vertical {
                cursor: row-resize;
            }
            #container1 { grid-area: 1 / 1 / 2 / 4; }
            #container2 { grid-area: 3 / 1 / 4 / 2; }
            #container3 { grid-area: 3 / 3 / 4 / 4; }
            #gutter-horizontal { grid-area: 3 / 2 / 4 / 3; }
            #gutter-vertical { grid-area: 2 / 1 / 3 / 4; }
        """),
        Div(
            Div(create_attribute_animation_svg(), id="container1", cls="resizable"),
            Div(id="gutter-vertical", cls="gutter gutter-vertical"),
            Div(create_transform_animation_svg(), id="container2", cls="resizable"),
            Div(id="gutter-horizontal", cls="gutter gutter-horizontal"),
            Div(create_motion_animation_svg(), id="container3", cls="resizable"),
            cls="grid-container"
        ),
        Script("""
        let gridState = {
            rows: ['1fr', '10px', '1fr'],
            columns: ['1fr', '10px', '1fr']
        };

        function setupResize() {
            const grid = document.querySelector('.grid-container');
            const gutterVertical = document.getElementById('gutter-vertical');
            const gutterHorizontal = document.getElementById('gutter-horizontal');

            interact('.gutter-vertical').draggable({
                axis: 'y',
                listeners: {
                    move(event) {
                        const gridRect = grid.getBoundingClientRect();
                        const newTopHeight = event.clientY - gridRect.top;
                        const newBottomHeight = gridRect.height - newTopHeight - gutterVertical.offsetHeight;
                        
                        gridState.rows = [`${newTopHeight}px`, '10px', `${newBottomHeight}px`];
                        updateGridTemplate();
                    }
                }
            });

            interact('.gutter-horizontal').draggable({
                axis: 'x',
                listeners: {
                    move(event) {
                        const gridRect = grid.getBoundingClientRect();
                        const newLeftWidth = event.clientX - gridRect.left;
                        const newRightWidth = gridRect.width - newLeftWidth - gutterHorizontal.offsetWidth;
                        
                        gridState.columns = [`${newLeftWidth}px`, '10px', `${newRightWidth}px`];
                        updateGridTemplate();
                    }
                }
            });
        }

        function updateGridTemplate() {
            const grid = document.querySelector('.grid-container');
            grid.style.gridTemplateRows = gridState.rows.join(' ');
            grid.style.gridTemplateColumns = gridState.columns.join(' ');
            resizeAllContainers();
        }

        function resizeAllContainers() {
            ['container1', 'container2', 'container3'].forEach(id => {
                const container = document.getElementById(id);
                resizeSvgContainer(id, container.clientWidth, container.clientHeight);
            });
        }

        function resizeSvgContainer(containerId, width, height) {
            const container = document.getElementById(containerId);
            const svg = container.querySelector('svg');
            if (svg) {
                // Update SVG size
                svg.setAttribute('width', '100%');
                svg.setAttribute('height', '100%');
                
                // Update viewBox to maintain aspect ratio
                const viewBox = svg.viewBox.baseVal;
                const aspectRatio = viewBox.width / viewBox.height;
                
                if (width / height > aspectRatio) {
                    viewBox.width = height * aspectRatio;
                    viewBox.height = height;
                } else {
                    viewBox.width = width;
                    viewBox.height = width / aspectRatio;
                }
                
                svg.setAttribute('viewBox', `0 0 ${viewBox.width} ${viewBox.height}`);
                svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
            }
        }

        function handleWindowResize() {
            const grid = document.querySelector('.grid-container');
            const gutterVertical = document.getElementById('gutter-vertical');
            const gutterHorizontal = document.getElementById('gutter-horizontal');

            // Reset to default layout if any dimension is in pixels
            if (gridState.rows.some(r => r.endsWith('px')) || gridState.columns.some(c => c.endsWith('px'))) {
                gridState.rows = ['1fr', '10px', '1fr'];
                gridState.columns = ['1fr', '10px', '1fr'];
            }

            updateGridTemplate();
        }

        setupResize();
        handleWindowResize(); // Initial call to set up the layout
        window.addEventListener('resize', handleWindowResize);
        document.body.addEventListener('htmx:afterSwap', () => {
            setupResize();
            handleWindowResize();
        });
        """)
    )

def create_attribute_animation_svg():
    animate = Animate(
        attributeName="cx",
        from_="0",
        to="100%",
        dur="5s",
        repeatCount="indefinite"
    )
    circle = Circle(
        15, 0, 50,  # r, cx, cy
        fill="blue",
        stroke="black",
        stroke_width=1,
    )
    animated_circle = circle(*[animate])
    
    return Svg(
        Title("Attribute Animation with SMIL"),
        Rect(width="100%", height="100%", x=0, y=0, stroke="black", stroke_width=1),
        animated_circle,
        width="100%", height="100%",
        viewBox="0 0 300 100",
        preserveAspectRatio="xMidYMid meet"
    )

def create_transform_animation_svg():
    animate_transform = AnimateTransform(
        attributeName="transform",
        begin="0s",
        dur="20s",
        type="rotate",
        from_="0 50 50",
        to="360 50 50",
        repeatCount="indefinite"
    )
    
    rect = Rect(
        width=30, height=30, x=35, y=35,
        fill="blue",
        stroke="black",
        stroke_width=1,
    )
    
    animated_rect = rect(*[animate_transform])
    
    return Svg(
        Title("SVG SMIL Animate with transform"),
        Rect(width="100%", height="100%", x=0, y=0, stroke="black", stroke_width=1),
        animated_rect,
        width="100%", height="100%",
        viewBox="0 0 100 100",
        preserveAspectRatio="xMidYMid meet"
    )

def create_motion_animation_svg():
    animate_motion = AnimateMotion(
        path="M 20,50 C 20,20 80,20 80,50 C 80,80 20,80 20,50 Z",
        dur="5s",
        repeatCount="indefinite",
        rotate="auto"
    )
    
    rect = Rect(
        width=20, height=20, x=-10, y=-10,
        fill="blue",
        stroke="black",
        stroke_width=1,
    )
    
    animated_rect = rect(*[animate_motion])
    
    return Svg(
        Title("SVG SMIL Animate with Path - Curved Motion"),
        Rect(width="100%", height="100%", x=0, y=0, stroke="black", stroke_width=1),
        animated_rect,
        width="100%", height="100%",
        viewBox="0 0 100 100",
        preserveAspectRatio="xMidYMid meet"
    )
    
# Run the app
serve()