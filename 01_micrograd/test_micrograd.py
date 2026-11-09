import math
from micrograd import Value

def close(x, y): return math.isclose(x, y, abs_tol=1e-9)

def test_add_same_value():
    a = Value(3.0); b = a + a; b.backward()
    assert close(a.grad, 2.0)

def test_paper_example():
    a, b, c, f = Value(2.0), Value(-3.0), Value(10.0), Value(-2.0)
    L = (a*b + c) * f; L.backward()
    assert (a.grad, b.grad, c.grad, f.grad) == (6.0, -4.0, -2.0, 4.0)

def test_matches_pytorch():
    import torch   # pip install torch
    vals = (2.0, 0.0, -3.0, 1.0, 6.8813735870195432)
    mine = [Value(v) for v in vals]
    x1, x2, w1, w2, b = mine
    (x1*w1 + x2*w2 + b).tanh().backward()
    ts = [torch.tensor(v, dtype=torch.double, requires_grad=True) for v in vals]
    t1, t2, tw1, tw2, tb = ts
    torch.tanh(t1*tw1 + t2*tw2 + tb).backward()
    for m, t in zip(mine, ts):
        assert close(m.grad, t.grad.item())

test_add_same_value(); test_paper_example(); test_matches_pytorch()
print("all tests passed")