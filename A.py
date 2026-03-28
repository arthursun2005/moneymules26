# import tifffile
import numpy as np
import jax
import jax.numpy as jnp
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt
import optax
import random
from tqdm import tqdm
import numpy as np
import gc
import traceback

IMG_SIZE = 3600

class EMA:
    def __init__(self, u=1e-2):
        self.x = 0.0
        self.w = 0.0
        self.u = u

    def add(self, x, w=1.0):
        u = self.u
        self.x = self.x * (1 - u) + x * w * u
        self.w = self.w * (1 - u) + w * u
        return self.x / self.w

    def get(self):
        return self.x / self.w

paths = glob('data/*dem.tif')

def eye_make_params(key, n_out, *, k):
    key, *subkeys = jax.random.split(key, 99)
    subkeys = iter(subkeys)

    def conv(key, shape):
        return jax.random.normal(key, (shape[0], shape[1], shape[1], shape[2], shape[3]))

    W, H = 3, 3
    M = [1, 2, 4, 8]
    params = {
        'rW4': jax.random.normal(next(subkeys), (k, M[3], M[2])),
        'rC4': conv(next(subkeys), (k, M[3], W, H)),
        'rb4': jnp.zeros((k, M[3],)),

        'rW3': jax.random.normal(next(subkeys), (k, M[2], M[1])),
        'rC3': conv(next(subkeys), (k, M[2], W, H)),
        'rb3': jnp.zeros((k, M[2],)),

        'rW2': jax.random.normal(next(subkeys), (k, M[1], M[0])),
        'rC2': conv(next(subkeys), (k, M[1], W, H)),
        'rb2': jnp.zeros((k, M[1],)),

        'rW1': jax.random.normal(next(subkeys), (k, M[0], 1)),
        'rC1': conv(next(subkeys), (k, M[0], W, H)),
        'rb1': jnp.zeros((k, M[0],)),

        'S': jnp.zeros((k, 1,)),
    }
    return key, params


def pool(x):
    # x: (B, W, H, C)
    a = (x.shape[1] // 2) * 2
    b = (x.shape[2] // 2) * 2
    x = x[:, :a, :b, :]
    return x.reshape(x.shape[0], a // 2, 2, b // 2, 2, x.shape[-1]).mean((2, 4))


def rpool(x):
    # x: (B, W, H, C)
    a = x.shape[1]
    b = x.shape[2]
    x = x[:, :, None, :, None, :]
    x = jnp.tile(x, (1, 1, 2, 1, 2, 1))
    return x.reshape(x.shape[0], a * 2, b * 2, x.shape[-1])


@jax.jit
def eye(key, params, x, *, training):
    keys = jax.random.split(key, params['S'].shape[0])
    def head(x, k,
             rW1, rC1, rb1, rW2, rC2, rb2, rW3, rC3, rb3, rW4, rC4, rb4, S):

        def rconv(x, w):
            return jax.lax.conv_general_dilated(x, w, (1, 1),
                                                 padding=[(1, 1), (1, 1)],
                                                dimension_numbers=('NHWC', 'OIHW', 'NHWC')) / (w.shape[1] * w.shape[2] * w.shape[3])


        act = jax.nn.leaky_relu
        V = x

        V = rpool(V)
        V = rconv(V, rC4)
        V = act(V + rb4)
        V = V @ rW4 / rW4.shape[0] ** 0.5

        V = rpool(V)
        V = rconv(V, rC3)
        V = act(V + rb3)
        V = V @ rW3 / rW3.shape[0] ** 0.5

        V = rpool(V)
        V = rconv(V, rC2)
        V = act(V + rb2)
        V = V @ rW2 / rW2.shape[0] ** 0.5

        V = rpool(V)
        V = rconv(V, rC1)
        V = act(V + rb1)
        V = V @ rW1 / rW1.shape[0] ** 0.5
        V = V.reshape(V.shape[0], V.shape[1], V.shape[2])

        return V

    return jax.vmap(head, (None, 0) + (0,) * len(params), 0)(
        x, keys,

        params['rW1'],
        params['rC1'],
        params['rb1'],
        params['rW2'],
        params['rC2'],
        params['rb2'],
        params['rW3'],
        params['rC3'],
        params['rb3'],
        params['rW4'],
        params['rC4'],
        params['rb4'],
        jnp.exp(params['S']),
    )

def make_x(p):
    p = p.split('_')[1]
    a = int(p[1:3])
    if p[0] == 'S': a = -a
    b = int(p[4:7])
    if p[3] == 'E': b = -b
    return [np.sin(a / 180 * np.pi), np.sin(b / 180 * np.pi)]

@jax.jit
def train_once(key, params, opt_state, x, y):
    key, subkey = jax.random.split(key)
    def compute(params):
        p = eye(subkey, params, x, training=True)
        err = (p - y[None, :]) ** 2
        return err.mean()
    loss, grads = jax.value_and_grad(compute)(params)
    updates, opt_state = opt.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return key, params, opt_state, loss

BS = 4
BB = 1
key = jax.random.PRNGKey(1)
key, params = eye_make_params(key, 1, k=1)
steps = 24000
opt = optax.adabelief(optax.cosine_onecycle_schedule(steps, 1e-2, 0.1), b1=0.88, b2=0.99)
opt_state = opt.init(params)
ema_loss = EMA()

for k in (bar := tqdm(range(steps))):
    try:
        idx = random.choices(paths, k=BS * BB)
        images = np.array([
            np.array(Image.open(p))[:IMG_SIZE, :IMG_SIZE] for p in idx
        ], dtype=np.float32) / 1e4
        assert images.shape == (BS * BB, IMG_SIZE, IMG_SIZE)
        X = np.array([
            make_x(p) for p in idx
        ])

        images = jnp.array(images)
        # X = jnp.array(X)

        xx, yy = np.meshgrid(np.arange(225) / 225, np.arange(225) / 225)
        T = (BS, 225, 225)
        X = np.concatenate(
            [
            np.broadcast_to(np.cos(xx / IMG_SIZE)[None, :, :, None], T + (1,)),
                np.broadcast_to(np.cos(yy / IMG_SIZE)[None, :, :, None], T + (1,)),
                np.broadcast_to(np.cos(xx / IMG_SIZE * 0.5)[None, :, :, None], T + (1,)),
                np.broadcast_to(np.cos(yy / IMG_SIZE * 0.5)[None, :, :, None], T + (1,)),
                np.broadcast_to(np.cos(xx / IMG_SIZE * 0.2)[None, :, :, None], T + (1,)),
                np.broadcast_to(np.cos(yy / IMG_SIZE * 0.2)[None, :, :, None], T + (1,)),
                np.broadcast_to(X[:, None, None, :], T + (2,)),
            ],
            3
        )
        print(X)

        print(images.shape, X.shape, images.max())

        # key, params, opt_state, loss = train_once(key, params, opt_state, jnp.ones((BS, 225, 225, 8)), images)
        key, params, opt_state, loss = train_once(key, params, opt_state, X, images)
        ema_loss.add(loss.item())

        # U = jnp.ones((BS, 225, 225, 16))
        # p = eye(key, params, U, training=True)
        # print(p)

        print(ema_loss.get())

        gc.collect()

        if k % 10 == 0:
            jnp.savez('params.npz', **params)
    except KeyboardInterrupt:
        print('wow!')
        break
    except:
        print('oops!')
        print(traceback.format_exc())
