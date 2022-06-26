"""The main entrypoint for linformer."""
import time

import typer
import keras
import numpy as np

from linformer.attention import LinearMultiHeadAttention


def run_experiment(
    sequence_length: int = 2000,
    batch_size: int = 25,
    embedding_dim: int = 256,
    projection_dim: int = 64,
    num_heads: int = 8,
    hidden_dim: int = 128,
) -> None:
    """Run a prediction time experiment."""

    # layers
    inp = keras.Input(shape=(sequence_length,))
    emb = keras.layers.Embedding(100, embedding_dim)(inp)
    lmha = LinearMultiHeadAttention(
        num_heads,
        hidden_dim,
        projection_dim=projection_dim)(emb, emb)

    # build model
    fast_model = keras.Model(inp, lmha)
    fast_model.summary()

    # set up data input
    data = np.random.choice(100, size=(batch_size, sequence_length))

    # run timed experiment
    t0 = time.time()
    _ = fast_model.predict(data)
    t1 = time.time()
    print(
        f"SEQUENCE LENGTH: {sequence_length}\n"
        f"BATCH SIZE:      {batch_size}\n"
        f"INFERENCE TIME:  {t1 - t0}"
    )


def main() -> None:
    typer.run(run_experiment)


if __name__ == "__main__":
    main()
