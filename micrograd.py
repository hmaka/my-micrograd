import math


class Value:

    def __init__(self, data: float, prev=(), op=""):
        self.data = data
        self.grad = 0.0
        self.prev = set(prev)
        self.op = op

    def __repr__(self):
        return f"Value(data={self.data})"
    

    def __add__(self, other):
        return Value(self.data + other.data, (self,other), "+")

    def __radd__(self,other):
        return self + other

    def __sub__(self, other):
        return Value(self.data - other.data, (self,other), "-")

    def __neg__(self):
        return Value(-self.data, (self,), "-")
    
    def __mul__(self, other):
        return Value(self.data * other.data, (self,other), "*")

    def __truediv__(self,other):
        return Value(self.data / other.data, (self,other), "/")
    
    def __pow__(self,other):
        return Value(self.data**other.data,(self,other), "**")

    def exp(self):
        return Value(math.exp(self.data), (self,), "exp")

    def log(self):
        return Value(math.log(self.data), (self,), "log")

    def backward(self):

        match self.op:
            case "+":
                first,second = self.prev
                # self = first + second

                first.grad += 1.0 * self.grad
                second.grad += 1.0 * self.grad

            case "-" if len(self.prev) == 2:
                first,second = self.prev
                # self = first - second

                first.grad += 1.0 * self.grad
                second.grad += -1.0 * self.grad

            case "-" if len(self.prev) == 1:
                first = list(self.prev)[0]
                # self = - first 

                first.grad += -1.0 * self.grad

            case "*":
                first,second = self.prev
                # self = first * second

                first.grad += second.data * self.grad
                second.grad += first.data * self.grad

            case "/":
                first,second = self.prev
                # self = first / second

                first.grad += (1/second.data) * self.grad

                # self = -1first*(second**-2)
                second.grad += ( -first.data/(second.data**2) )  * self.grad

            case "**":
                first,second = self.prev
                # self = first ** second

                first.grad += second.data*(first.data**(second.data-1)) * self.grad

                # dself/dsecond = (first ** second) * ln(second) 
                # note first ** second = self
                second.grad += self * math.exp(second) * self.grad

            case "exp":
                first = list(self.prev)[0]
                # self = e**first 

                first.grad += self.data * self.grad

            case "log":
                first = list(self.prev)[0]
                # self = log(first) 
                first.grad += 1/first.data * self.grad
                

            case _: raise ValueError(f"Uknown op: {self.op}")

    
# This function updates the grads of all the weights in the Value graph    
def back_prop(root: Value):
    # set start grad to 1.0
    root.grad = 1.0 

    def top_sort(root: Value, seen, ordering):
        
        seen.add(root)

        for p in root.prev:
            if p not in seen:
                top_sort(p, seen, ordering)

        ordering.append(root)

        return ordering

    
    reverse_top_ordering = reversed(top_sort(root, set(), []))

     
























