import z3
from utils import create_tensor, equalize_tensors

def test_z3_sanity():
    shape = (2,2)
    
    t1 = create_tensor(shape, "t1")
    t2 = create_tensor(shape, "t2")
    
    t1_eq_t2 = equalize_tensors([t1,t2])
    t1_neq_t2 = z3.Not(z3.And(t1_eq_t2))
    
    solver = z3.Solver()
    solver.add(t1_eq_t2)
    result = solver.check()
    print("Expect: sat")
    print(result)
    
    solver.add(t1_neq_t2)
    result = solver.check()
    print("Expect: unsat")
    print(result)
    
if __name__ == "__main__":
    test_z3_sanity()