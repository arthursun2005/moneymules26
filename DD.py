import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

import numpy as np
import jax
import jax.numpy as jnp
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt

loaded_npz = np.load('params.npz')
params = {k: jnp.array(loaded_npz[k]) for k in loaded_npz.files}

IMG_SIZE = 3600

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
             rW1, rC1, rb1, rW2, rC2, rb2, rW3, rC3, rb3, rW4, rC4, rb4,
             W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6, W7, b7,
             S):

        def rconv(x, w):
            return jax.lax.conv_general_dilated(x, w, (1, 1),
                                                 padding=[(3, 3), (3, 3)],
                                                dimension_numbers=('NHWC', 'OIHW', 'NHWC')) / (w.shape[1] * w.shape[2] * w.shape[3]) ** 0.5


        act = jax.nn.swish

        x = x @ W1 / W1.shape[0] ** 0.5
        x = act(x + b1)

        x = x @ W2 / W2.shape[0] ** 0.5
        x = act(x + b2)

        x = x @ W3 / W3.shape[0] ** 0.5
        x = act(x + b3)

        # x = x @ W4 / W4.shape[0] ** 0.5
        # x = act(x + b4)
        #
        # x = x @ W5 / W5.shape[0] ** 0.5
        # x = act(x + b5)

        # x = x @ W6 / W6.shape[0] ** 0.5
        # x = act(x + b6)
        #
        # x = x @ W7 / W7.shape[0] ** 0.5
        # x = act(x + b7)

        V = x

        V = rpool(V)
        V = rconv(V, rC4)
        V = act(V + rb4)
        V = V @ rW4 / rW4.shape[0] ** 0.5

        # V = V @ W6 / W6.shape[0] ** 0.5
        # V = act(V + b6)
        #
        # V = V @ W7 / W7.shape[0] ** 0.5
        # V = act(V + b7)

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

        V = V @ W6 / W6.shape[0] ** 0.5
        V = act(V + b6)

        V = V @ W7 / W7.shape[0] ** 0.5
        V = act(V + b7)

        # V = V.reshape(V.shape[0], V.shape[1], V.shape[2])

        return V.mean(-1)

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

        params['W1'],
        params['b1'],
        params['W2'],
        params['b2'],
        params['W3'],
        params['b3'],
        params['W4'],
        params['b4'],
        params['W5'],
        params['b5'],
        params['W6'],
        params['b6'],
        params['W7'],
        params['b7'],

        jnp.exp(params['S']),
    )

# path = glob('data/*dem.tif')[0]
path = sorted(glob('data/*dem.tif'))[4]

img = np.array(Image.open(path))[:IMG_SIZE, :IMG_SIZE]

def make_x(p):
    p = p.split('_')[1]
    a = int(p[1:3])
    if p[0] == 'S': a = -a
    b = int(p[4:7])
    if p[3] == 'E': b = -b
    return [np.sin(a / 180 * np.pi), np.sin(b / 180 * np.pi)]

plt.imshow(img)
plt.show()

X = np.array([make_x(path)])

BS = 1

xx, yy = np.meshgrid(np.arange(225) / 225, np.arange(225) / 225)
T = (BS, 225, 225)
# X = np.concatenate(
#     [
#         np.broadcast_to(np.cos(xx / IMG_SIZE)[None, :, :, None], T + (1,)),
#         np.broadcast_to(np.cos(yy / IMG_SIZE)[None, :, :, None], T + (1,)),
#         np.broadcast_to(np.cos(xx / IMG_SIZE / 0.5)[None, :, :, None], T + (1,)),
#         np.broadcast_to(np.cos(yy / IMG_SIZE / 0.5)[None, :, :, None], T + (1,)),
#         np.broadcast_to(np.cos(xx / IMG_SIZE / 0.2)[None, :, :, None], T + (1,)),
#         np.broadcast_to(np.cos(yy / IMG_SIZE / 0.2)[None, :, :, None], T + (1,)),
#         np.broadcast_to(X[:, None, None, :], T + (2,)),
#
#         np.broadcast_to(np.cos(xx / IMG_SIZE)[None, :, :, None], T + (1,)),
#         np.broadcast_to(np.cos(yy / IMG_SIZE)[None, :, :, None], T + (1,)),
#         np.broadcast_to(np.cos(xx / IMG_SIZE / 0.15)[None, :, :, None], T + (1,)),
#         np.broadcast_to(np.cos(yy / IMG_SIZE / 0.15)[None, :, :, None], T + (1,)),
#         np.broadcast_to(np.cos(xx / IMG_SIZE / 0.1)[None, :, :, None], T + (1,)),
#         np.broadcast_to(np.cos(yy / IMG_SIZE / 0.1)[None, :, :, None], T + (1,)),
#         np.broadcast_to(X[:, None, None, :], T + (2,)),
#
#         np.broadcast_to(np.cos(xx / IMG_SIZE)[None, :, :, None], T + (1,)),
#         np.broadcast_to(np.cos(yy / IMG_SIZE)[None, :, :, None], T + (1,)),
#         np.broadcast_to(np.cos(xx / IMG_SIZE / 0.05)[None, :, :, None], T + (1,)),
#         np.broadcast_to(np.cos(yy / IMG_SIZE / 0.05)[None, :, :, None], T + (1,)),
#         np.broadcast_to(np.cos(xx / IMG_SIZE / 0.02)[None, :, :, None], T + (1,)),
#         np.broadcast_to(np.cos(yy / IMG_SIZE / 0.02)[None, :, :, None], T + (1,)),
#         np.broadcast_to(X[:, None, None, :], T + (2,)),
#
#         np.broadcast_to(np.cos(xx / IMG_SIZE)[None, :, :, None], T + (1,)),
#         np.broadcast_to(np.cos(yy / IMG_SIZE)[None, :, :, None], T + (1,)),
#         np.broadcast_to(np.cos(xx / IMG_SIZE * 0.5)[None, :, :, None], T + (1,)),
#         np.broadcast_to(np.cos(yy / IMG_SIZE * 0.5)[None, :, :, None], T + (1,)),
#         np.broadcast_to(np.cos(xx / IMG_SIZE * 0.2)[None, :, :, None], T + (1,)),
#         np.broadcast_to(np.cos(yy / IMG_SIZE * 0.2)[None, :, :, None], T + (1,)),
#         np.broadcast_to(X[:, None, None, :], T + (2,)),
#
#         np.broadcast_to(np.cos(xx / IMG_SIZE)[None, :, :, None], T + (1,)),
#         np.broadcast_to(np.cos(yy / IMG_SIZE)[None, :, :, None], T + (1,)),
#         np.broadcast_to(np.cos(xx / IMG_SIZE / 0.5)[None, :, :, None], T + (1,)),
#         np.broadcast_to(np.cos(yy / IMG_SIZE / 0.5)[None, :, :, None], T + (1,)),
#         np.broadcast_to(np.cos(xx / IMG_SIZE / 0.2)[None, :, :, None], T + (1,)),
#         np.broadcast_to(np.cos(yy / IMG_SIZE / 0.2)[None, :, :, None], T + (1,)),
#         np.broadcast_to(X[:, None, None, :], T + (2,)),
#
#         np.broadcast_to(np.cos(xx / IMG_SIZE)[None, :, :, None], T + (1,)),
#         np.broadcast_to(np.cos(yy / IMG_SIZE)[None, :, :, None], T + (1,)),
#         np.broadcast_to(np.cos(xx / IMG_SIZE / 0.15)[None, :, :, None], T + (1,)),
#         np.broadcast_to(np.cos(yy / IMG_SIZE / 0.15)[None, :, :, None], T + (1,)),
#         np.broadcast_to(np.cos(xx / IMG_SIZE / 0.12)[None, :, :, None], T + (1,)),
#         np.broadcast_to(np.cos(yy / IMG_SIZE / 0.12)[None, :, :, None], T + (1,)),
#         np.broadcast_to(X[:, None, None, :], T + (2,)),
#
#         np.broadcast_to(np.cos(xx / IMG_SIZE)[None, :, :, None], T + (1,)),
#         np.broadcast_to(np.cos(yy / IMG_SIZE)[None, :, :, None], T + (1,)),
#         np.broadcast_to(np.cos(xx / IMG_SIZE / 0.05)[None, :, :, None], T + (1,)),
#         np.broadcast_to(np.cos(yy / IMG_SIZE / 0.05)[None, :, :, None], T + (1,)),
#         np.broadcast_to(np.cos(xx / IMG_SIZE / 0.02)[None, :, :, None], T + (1,)),
#         np.broadcast_to(np.cos(yy / IMG_SIZE / 0.02)[None, :, :, None], T + (1,)),
#         np.broadcast_to(X[:, None, None, :], T + (2,)),
#
#         np.broadcast_to(np.cos(xx / IMG_SIZE)[None, :, :, None], T + (1,)),
#         np.broadcast_to(np.cos(yy / IMG_SIZE)[None, :, :, None], T + (1,)),
#         np.broadcast_to(np.cos(xx / IMG_SIZE * 0.5)[None, :, :, None], T + (1,)),
#         np.broadcast_to(np.cos(yy / IMG_SIZE * 0.5)[None, :, :, None], T + (1,)),
#         np.broadcast_to(np.cos(xx / IMG_SIZE * 0.2)[None, :, :, None], T + (1,)),
#         np.broadcast_to(np.cos(yy / IMG_SIZE * 0.2)[None, :, :, None], T + (1,)),
#         np.broadcast_to(X[:, None, None, :], T + (2,)),
#     ],
#     3
# )

X = np.concatenate(
    [
        np.broadcast_to(np.cos(xx / IMG_SIZE)[None, :, :, None], T + (1,)),
        np.broadcast_to(np.cos(yy / IMG_SIZE)[None, :, :, None], T + (1,)),
        np.broadcast_to(np.cos(xx / IMG_SIZE / 0.5)[None, :, :, None], T + (1,)),
        np.broadcast_to(np.cos(yy / IMG_SIZE / 0.5)[None, :, :, None], T + (1,)),
        np.broadcast_to(np.cos(xx / IMG_SIZE / 0.2)[None, :, :, None], T + (1,)),
        np.broadcast_to(np.cos(yy / IMG_SIZE / 0.2)[None, :, :, None], T + (1,)),
        np.broadcast_to(X[:, None, None, :], T + (2,)),

        np.broadcast_to(np.cos(xx / IMG_SIZE)[None, :, :, None], T + (1,)),
        np.broadcast_to(np.cos(yy / IMG_SIZE)[None, :, :, None], T + (1,)),
        np.broadcast_to(np.cos(xx / IMG_SIZE / 0.15)[None, :, :, None], T + (1,)),
        np.broadcast_to(np.cos(yy / IMG_SIZE / 0.15)[None, :, :, None], T + (1,)),
        np.broadcast_to(np.cos(xx / IMG_SIZE / 0.1)[None, :, :, None], T + (1,)),
        np.broadcast_to(np.cos(yy / IMG_SIZE / 0.1)[None, :, :, None], T + (1,)),
        np.broadcast_to(X[:, None, None, :], T + (2,)),

        # np.broadcast_to(np.cos(xx / IMG_SIZE)[None, :, :, None], T + (1,)),
        # np.broadcast_to(np.cos(yy / IMG_SIZE)[None, :, :, None], T + (1,)),
        # np.broadcast_to(np.cos(xx / IMG_SIZE / 0.05)[None, :, :, None], T + (1,)),
        # np.broadcast_to(np.cos(yy / IMG_SIZE / 0.05)[None, :, :, None], T + (1,)),
        # np.broadcast_to(np.cos(xx / IMG_SIZE / 0.02)[None, :, :, None], T + (1,)),
        # np.broadcast_to(np.cos(yy / IMG_SIZE / 0.02)[None, :, :, None], T + (1,)),
        # np.broadcast_to(X[:, None, None, :], T + (2,)),
        #
        # np.broadcast_to(np.cos(xx / IMG_SIZE)[None, :, :, None], T + (1,)),
        # np.broadcast_to(np.cos(yy / IMG_SIZE)[None, :, :, None], T + (1,)),
        # np.broadcast_to(np.cos(xx / IMG_SIZE * 0.5)[None, :, :, None], T + (1,)),
        # np.broadcast_to(np.cos(yy / IMG_SIZE * 0.5)[None, :, :, None], T + (1,)),
        # np.broadcast_to(np.cos(xx / IMG_SIZE * 0.2)[None, :, :, None], T + (1,)),
        # np.broadcast_to(np.cos(yy / IMG_SIZE * 0.2)[None, :, :, None], T + (1,)),
        # np.broadcast_to(X[:, None, None, :], T + (2,)),

        np.broadcast_to(X[:, None, None, :], T + (2,)),
        np.broadcast_to(X[:, None, None, :], T + (2,)),
        np.broadcast_to(X[:, None, None, :], T + (2,)),
        np.broadcast_to(X[:, None, None, :], T + (2,)),

        np.broadcast_to(X[:, None, None, :], T + (2,)),
        np.broadcast_to(X[:, None, None, :], T + (2,)),
        np.broadcast_to(X[:, None, None, :], T + (2,)),
        np.broadcast_to(X[:, None, None, :], T + (2,)),

        np.broadcast_to(np.cos(xx / IMG_SIZE)[None, :, :, None], T + (1,)),
        np.broadcast_to(np.cos(yy / IMG_SIZE)[None, :, :, None], T + (1,)),
        np.broadcast_to(np.cos(xx / IMG_SIZE / 0.5)[None, :, :, None], T + (1,)),
        np.broadcast_to(np.cos(yy / IMG_SIZE / 0.5)[None, :, :, None], T + (1,)),
        np.broadcast_to(np.cos(xx / IMG_SIZE / 0.2)[None, :, :, None], T + (1,)),
        np.broadcast_to(np.cos(yy / IMG_SIZE / 0.2)[None, :, :, None], T + (1,)),
        np.broadcast_to(X[:, None, None, :], T + (2,)),

        # np.broadcast_to(np.cos(xx / IMG_SIZE)[None, :, :, None], T + (1,)),
        # np.broadcast_to(np.cos(yy / IMG_SIZE)[None, :, :, None], T + (1,)),
        # np.broadcast_to(np.cos(xx / IMG_SIZE / 0.15)[None, :, :, None], T + (1,)),
        # np.broadcast_to(np.cos(yy / IMG_SIZE / 0.15)[None, :, :, None], T + (1,)),
        # np.broadcast_to(np.cos(xx / IMG_SIZE / 0.12)[None, :, :, None], T + (1,)),
        # np.broadcast_to(np.cos(yy / IMG_SIZE / 0.12)[None, :, :, None], T + (1,)),
        # np.broadcast_to(X[:, None, None, :], T + (2,)),

        np.broadcast_to(np.cos(xx / IMG_SIZE)[None, :, :, None], T + (1,)),
        np.broadcast_to(np.cos(yy / IMG_SIZE)[None, :, :, None], T + (1,)),
        np.broadcast_to(np.cos(xx / IMG_SIZE / 0.05)[None, :, :, None], T + (1,)),
        np.broadcast_to(np.cos(yy / IMG_SIZE / 0.05)[None, :, :, None], T + (1,)),
        np.broadcast_to(np.cos(xx / IMG_SIZE / 0.02)[None, :, :, None], T + (1,)),
        np.broadcast_to(np.cos(yy / IMG_SIZE / 0.02)[None, :, :, None], T + (1,)),
        np.broadcast_to(X[:, None, None, :], T + (2,)),

        np.broadcast_to(X[:, None, None, :], T + (2,)),
        np.broadcast_to(X[:, None, None, :], T + (2,)),
        np.broadcast_to(X[:, None, None, :], T + (2,)),
        np.broadcast_to(X[:, None, None, :], T + (2,)),

        # np.broadcast_to(np.cos(xx / IMG_SIZE)[None, :, :, None], T + (1,)),
        # np.broadcast_to(np.cos(yy / IMG_SIZE)[None, :, :, None], T + (1,)),
        # np.broadcast_to(np.cos(xx / IMG_SIZE * 0.5)[None, :, :, None], T + (1,)),
        # np.broadcast_to(np.cos(yy / IMG_SIZE * 0.5)[None, :, :, None], T + (1,)),
        # np.broadcast_to(np.cos(xx / IMG_SIZE * 0.2)[None, :, :, None], T + (1,)),
        # np.broadcast_to(np.cos(yy / IMG_SIZE * 0.2)[None, :, :, None], T + (1,)),

        np.broadcast_to(X[:, None, None, :], T + (2,)),
        np.broadcast_to(X[:, None, None, :], T + (2,)),
        np.broadcast_to(X[:, None, None, :], T + (2,)),
        np.broadcast_to(X[:, None, None, :], T + (2,)),
    ],
    3
)

key = jax.random.PRNGKey(0)
P = eye(key, params, X, training=True)

print(P.shape)
print(P * 1e4)
print(img)

# P = np.maximum(0, np.array(P)[0, 0, :, :])
P = np.array(P)[0, 0, :, :]
print(np.abs(P * 1e4 - img).mean())
print(np.abs(img.mean() - img).mean())

plt.imshow(P)
plt.show()
