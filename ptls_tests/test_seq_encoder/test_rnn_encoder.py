import torch

from ptls.seq_encoder.rnn_encoder import RnnEncoder
from ptls.trx_encoder import PaddedBatch


def get_data():
    return PaddedBatch(
        payload=torch.arange(4*5*8).view(4, 8, 5).float(),
        length=torch.tensor([4, 2, 6, 8])
    )


def test_example():
    model = RnnEncoder(
        input_size=5,
        hidden_size=6,
        is_reduce_sequence=False,
    )

    x = get_data()

    out = model(x)
    assert out.payload.shape == (4, 8, 6)


def test_last_step():
    model = RnnEncoder(
        input_size=5,
        hidden_size=6,
        is_reduce_sequence=True,
    )

    x = get_data()

    h = model(x)
    assert h.shape == (4, 6)
