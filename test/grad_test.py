import math

from micrograd import Value, back_prop


def calc_num_grads(values, f, h):
    grads = []

    for i, val in enumerate(values):

        values_add_h = [v for v in values]
        values_add_h[i] += h

        values_minus_h = [v for v in values]
        values_minus_h[i] -= h

        grad = (f(*values_add_h) - f(*values_minus_h)) / (2*h)

        grads.append(grad)
    
    return grads

def calc_val_grads(raw_numbers, f):
    values = [Value(n) for n in raw_numbers]

    # forward then backward pass
    back_prop(f(*values))
    value_grads = [v.grad for v in values]
    return value_grads

def test_add():
    h = 0.0001
    a,b = 1,2

    def f(a,b):
        return a + b

    raw = [a,b]
    num_grads = calc_num_grads(raw, f, h)
    value_grads =  calc_val_grads(raw, f)


    for grad1, grad2 in zip(value_grads,num_grads):
        assert abs(grad1 - grad2) < 1e-5, f"Not OK, backwards computed:{grad1}, expected: {grad2}"

def test_mul():
    h = 0.0001
    a,b = 18,57

    def f(a,b):
        return a * b

    raw = [a,b]
    num_grads = calc_num_grads(raw, f, h)
    value_grads =  calc_val_grads(raw, f)


    for grad1, grad2 in zip(value_grads,num_grads):
        assert abs(grad1 - grad2) < 1e-5, f"Not OK, backwards computed:{grad1}, expected: {grad2}"

def test_sub():
    h = 0.0001
    a,b = 5,3

    def f(a,b):
        return a - b

    raw = [a,b]
    num_grads = calc_num_grads(raw, f, h)
    value_grads = calc_val_grads(raw, f)

    for grad1, grad2 in zip(value_grads, num_grads):
        assert abs(grad1 - grad2) < 1e-5, f"Not OK, backwards computed:{grad1}, expected: {grad2}"

def test_neg():
    h = 0.0001
    a = 3

    def f(a):
        return -a

    raw = [a]
    num_grads = calc_num_grads(raw, f, h)
    value_grads = calc_val_grads(raw, f)

    for grad1, grad2 in zip(value_grads, num_grads):
        assert abs(grad1 - grad2) < 1e-5, f"Not OK, backwards computed:{grad1}, expected: {grad2}"

def test_div():
    h = 0.0001
    a,b = 6,2

    def f(a,b):
        return a / b

    raw = [a,b]
    num_grads = calc_num_grads(raw, f, h)
    value_grads = calc_val_grads(raw, f)

    for grad1, grad2 in zip(value_grads, num_grads):
        assert abs(grad1 - grad2) < 1e-5, f"Not OK, backwards computed:{grad1}, expected: {grad2}"

def test_pow():
    h = 0.0001
    a,b = 2,3

    def f(a,b):
        return a ** b

    raw = [a,b]
    num_grads = calc_num_grads(raw, f, h)
    value_grads = calc_val_grads(raw, f)

    for grad1, grad2 in zip(value_grads, num_grads):
        assert abs(grad1 - grad2) < 1e-5, f"Not OK, backwards computed:{grad1}, expected: {grad2}"

def test_exp():
    h = 0.0001
    a = 1

    def f(a):

        if isinstance(a,float): return math.exp(a)
        return a.exp()

    raw = [a]
    num_grads = calc_num_grads(raw, f, h)
    value_grads = calc_val_grads(raw, f)

    for grad1, grad2 in zip(value_grads, num_grads):
        assert abs(grad1 - grad2) < 1e-5, f"Not OK, backwards computed:{grad1}, expected: {grad2}"

def test_log():
    h = 0.0001
    a = 2

    def f(a):
        if isinstance(a,float): return math.log(a)
        return a.log()

    raw = [a]
    num_grads = calc_num_grads(raw, f, h)
    value_grads = calc_val_grads(raw, f)

    for grad1, grad2 in zip(value_grads, num_grads):
        assert abs(grad1 - grad2) < 1e-5, f"Not OK, backwards computed:{grad1}, expected: {grad2}"

def test_karpathy():
    h = 0.00001
    a,b,c = 2.0,3.0,4.0

    def f(a, b, c):
        b = math.exp(b) if isinstance(b,float) else b.exp()

        return -a**3 + 3*b - 1.0/c + b**2.5 - a**0.5

    raw = [a,b,c]
    num_grads = calc_num_grads(raw, f, h)
    value_grads = calc_val_grads(raw, f)

    for grad1, grad2 in zip(value_grads, num_grads):
        assert abs(grad1 - grad2) < 1e-5, f"Not OK, backwards computed:{grad1}, expected: {grad2}"
