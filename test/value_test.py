from micrograd import Value

def test_add():
    a = Value(1)
    b = Value(2)
    c = a + b
    assert c.data == 3
    assert c.prev == {a, b}
    assert c.op == "+"

def test_sub():
    a = Value(3)
    b = Value(1)
    c = a - b
    assert c.data == 2
    assert c.prev == {a, b}
    assert c.op == "-"

def test_mul():
    a = Value(2)
    b = Value(3)
    c = a * b
    assert c.data == 6
    assert c.prev == {a, b}
    assert c.op == "*"

def test_div():
    a = Value(6)
    b = Value(2)
    c = a / b
    assert c.data == 3
    assert c.prev == {a, b}
    assert c.op == "/"

def test_pow():
    a = Value(2)
    b = Value(3)
    c = a ** b
    assert c.data == 8
    assert c.prev == {a,b}
    assert c.op == "**"

def test_neg():
    a = Value(3)
    c = -a
    assert c.data == -3
    assert c.prev == {a}
    assert c.op == "-"

def test_exp():
    a = Value(0)
    c = a.exp()
    assert c.data == 1
    assert c.prev == {a}
    assert c.op == "exp"

def test_log():
    a = Value(1)
    c = a.log()
    assert c.data == 0
    assert c.prev == {a}
    assert c.op == "log"
