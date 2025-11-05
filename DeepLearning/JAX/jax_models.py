"""
JAX/Flax Deep Learning Models
High-performance numerical computing and neural networks with JAX
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime


class JAXModels:
    """Comprehensive JAX/Flax model implementations"""

    def __init__(self):
        """Initialize JAX models manager"""
        self.models = []

    def create_mlp_model(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create Multi-Layer Perceptron with Flax

        Args:
            model_config: Model configuration

        Returns:
            Model details
        """
        model = {
            'name': model_config.get('name', 'MLP'),
            'type': 'mlp',
            'features': model_config.get('features', [128, 64, 10]),
            'activation': model_config.get('activation', 'relu'),
            'created_at': datetime.now().isoformat()
        }

        code = f"""
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax

class MLP(nn.Module):
    features: list = {model['features']}

    @nn.compact
    def __call__(self, x, training: bool = False):
        for feat in self.features[:-1]:
            x = nn.Dense(feat)(x)
            x = nn.relu(x)
            if training:
                x = nn.Dropout(rate=0.5)(x, deterministic=not training)

        x = nn.Dense(self.features[-1])(x)
        return x

# Initialize model
model = MLP()
key = jax.random.PRNGKey(0)
variables = model.init(key, jnp.ones((1, 784)))

# Create optimizer and training state
tx = optax.adam(learning_rate=0.001)
state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=variables['params'],
    tx=tx
)

# Training step
@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn({{'params': params}}, batch['image'], training=True)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits, batch['label']
        ).mean()
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss
"""

        model['code'] = code
        self.models.append(model)

        print(f"✓ MLP model created: {model['name']}")
        print(f"  Features: {model['features']}")
        return model

    def create_cnn_model(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create CNN with Flax

        Args:
            model_config: Model configuration

        Returns:
            Model details
        """
        model = {
            'name': model_config.get('name', 'CNN'),
            'type': 'convolutional',
            'num_classes': model_config.get('num_classes', 10),
            'created_at': datetime.now().isoformat()
        }

        code = f"""
import jax
import jax.numpy as jnp
from flax import linen as nn

class CNN(nn.Module):
    num_classes: int = {model['num_classes']}

    @nn.compact
    def __call__(self, x, training: bool = False):
        # Conv block 1
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Conv block 2
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Conv block 3
        x = nn.Conv(features=128, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Fully connected
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        if training:
            x = nn.Dropout(rate=0.5)(x, deterministic=not training)

        x = nn.Dense(features=self.num_classes)(x)
        return x

# Initialize and use
model = CNN()
key = jax.random.PRNGKey(0)
variables = model.init(key, jnp.ones((1, 28, 28, 1)))
"""

        model['code'] = code
        self.models.append(model)

        print(f"✓ CNN model created: {model['name']}")
        print(f"  Classes: {model['num_classes']}")
        return model

    def create_transformer_model(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create Transformer with Flax

        Args:
            model_config: Model configuration

        Returns:
            Model details
        """
        model = {
            'name': model_config.get('name', 'Transformer'),
            'type': 'transformer',
            'num_heads': model_config.get('num_heads', 8),
            'num_layers': model_config.get('num_layers', 6),
            'd_model': model_config.get('d_model', 512),
            'created_at': datetime.now().isoformat()
        }

        code = f"""
import jax
import jax.numpy as jnp
from flax import linen as nn

class TransformerBlock(nn.Module):
    num_heads: int = {model['num_heads']}
    d_model: int = {model['d_model']}
    dim_feedforward: int = 2048

    @nn.compact
    def __call__(self, x, mask=None, training: bool = False):
        # Multi-head attention
        attn_output = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model
        )(x, x, mask=mask)

        # Add & Norm
        x = nn.LayerNorm()(x + attn_output)

        # Feed-forward
        ff_output = nn.Dense(self.dim_feedforward)(x)
        ff_output = nn.relu(ff_output)
        if training:
            ff_output = nn.Dropout(rate=0.1)(ff_output, deterministic=not training)
        ff_output = nn.Dense(self.d_model)(ff_output)

        # Add & Norm
        x = nn.LayerNorm()(x + ff_output)

        return x

class Transformer(nn.Module):
    num_layers: int = {model['num_layers']}
    num_heads: int = {model['num_heads']}
    d_model: int = {model['d_model']}
    vocab_size: int = 10000

    @nn.compact
    def __call__(self, x, mask=None, training: bool = False):
        # Embedding
        x = nn.Embed(num_embeddings=self.vocab_size, features=self.d_model)(x)

        # Positional encoding
        x = x + self.param('pos_encoding',
                           nn.initializers.normal(),
                           (1, x.shape[1], self.d_model))

        # Transformer blocks
        for _ in range(self.num_layers):
            x = TransformerBlock(
                num_heads=self.num_heads,
                d_model=self.d_model
            )(x, mask=mask, training=training)

        return x

model = Transformer()
"""

        model['code'] = code
        self.models.append(model)

        print(f"✓ Transformer model created: {model['name']}")
        print(f"  Layers: {model['num_layers']}, Heads: {model['num_heads']}")
        return model

    def create_training_utilities(self) -> str:
        """Generate JAX training utilities"""

        code = """
import jax
import jax.numpy as jnp
from flax.training import train_state, checkpoints
import optax

# Training state with metrics
class TrainState(train_state.TrainState):
    batch_stats: Any = None

# Loss functions
def cross_entropy_loss(logits, labels):
    one_hot_labels = jax.nn.one_hot(labels, logits.shape[-1])
    return optax.softmax_cross_entropy(logits, one_hot_labels).mean()

def mse_loss(predictions, targets):
    return jnp.mean((predictions - targets) ** 2)

# Metrics
def compute_metrics(logits, labels):
    loss = cross_entropy_loss(logits, labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return {'loss': loss, 'accuracy': accuracy}

# Learning rate schedules
def create_learning_rate_fn(
    base_learning_rate: float,
    steps_per_epoch: int,
    num_epochs: int,
    warmup_epochs: int = 5
):
    warmup_fn = optax.linear_schedule(
        init_value=0.,
        end_value=base_learning_rate,
        transition_steps=warmup_epochs * steps_per_epoch
    )

    cosine_epochs = max(num_epochs - warmup_epochs, 1)
    cosine_fn = optax.cosine_decay_schedule(
        init_value=base_learning_rate,
        decay_steps=cosine_epochs * steps_per_epoch
    )

    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[warmup_epochs * steps_per_epoch]
    )

    return schedule_fn

# Checkpointing
def save_checkpoint(state, workdir, step):
    checkpoints.save_checkpoint(
        ckpt_dir=workdir,
        target=state,
        step=step,
        overwrite=True
    )

def restore_checkpoint(state, workdir):
    return checkpoints.restore_checkpoint(
        ckpt_dir=workdir,
        target=state
    )
"""

        print("✓ Training utilities generated")
        return code

    def create_advanced_features(self) -> str:
        """Generate advanced JAX features"""

        code = """
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, pmap

# 1. Automatic differentiation
def loss_fn(params, x, y):
    predictions = model.apply(params, x)
    return jnp.mean((predictions - y) ** 2)

# Gradient function
grad_fn = jax.grad(loss_fn)
gradients = grad_fn(params, x_batch, y_batch)

# 2. JIT compilation for speed
@jax.jit
def fast_function(x):
    return jnp.dot(x, x.T)

# 3. Vectorization with vmap
def single_prediction(params, x):
    return model.apply(params, x)

# Vectorize across batch
batch_prediction = jax.vmap(single_prediction, in_axes=(None, 0))
predictions = batch_prediction(params, x_batch)

# 4. Parallelization with pmap (multi-GPU/TPU)
@jax.pmap
def parallel_train_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['image'])
        loss = cross_entropy_loss(logits, batch['label'])
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    grads = jax.lax.pmean(grads, axis_name='batch')
    state = state.apply_gradients(grads=grads)
    return state, loss

# 5. Random number generation
key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)
random_data = jax.random.normal(subkey, shape=(100, 784))

# 6. Control flow with lax
def conditional_fn(x):
    return jax.lax.cond(
        x > 0,
        lambda x: x ** 2,
        lambda x: -x,
        x
    )
"""

        print("✓ Advanced features code generated")
        return code

    def get_jax_info(self) -> Dict[str, Any]:
        """Get JAX manager information"""
        return {
            'models_created': len(self.models),
            'framework': 'JAX/Flax',
            'version': '0.4.20',
            'timestamp': datetime.now().isoformat()
        }


def demo():
    """Demonstrate JAX models"""

    print("=" * 60)
    print("JAX/Flax Deep Learning Models Demo")
    print("=" * 60)

    jax_models = JAXModels()

    print("\n1. Creating MLP model...")
    mlp = jax_models.create_mlp_model({
        'name': 'MultiLayerPerceptron',
        'features': [256, 128, 64, 10]
    })
    print(mlp['code'][:300] + "...\n")

    print("\n2. Creating CNN model...")
    cnn = jax_models.create_cnn_model({
        'name': 'ConvolutionalNetwork',
        'num_classes': 10
    })
    print(cnn['code'][:300] + "...\n")

    print("\n3. Creating Transformer model...")
    transformer = jax_models.create_transformer_model({
        'name': 'TransformerEncoder',
        'num_heads': 8,
        'num_layers': 6,
        'd_model': 512
    })
    print(transformer['code'][:300] + "...\n")

    print("\n4. Generating training utilities...")
    utilities = jax_models.create_training_utilities()
    print(utilities[:300] + "...\n")

    print("\n5. Generating advanced features...")
    advanced = jax_models.create_advanced_features()
    print(advanced[:300] + "...\n")

    print("\n6. JAX summary:")
    info = jax_models.get_jax_info()
    print(f"  Models created: {info['models_created']}")
    print(f"  Framework: {info['framework']} {info['version']}")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
