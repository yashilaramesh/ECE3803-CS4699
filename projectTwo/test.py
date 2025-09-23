import numpy as np
patterns = np.array([[1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 
  0, ],
 [1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0, 
  1, ],
 [1,  0,  1,  1,  0,  1,  1,  1,  0,  0,  0,  1,  1,  1,  0,  0,  1,  0,  1,  1,  1,  0,  1,  1, 
  1, ]])
for idx, target in enumerate(patterns, start=1):
    print(f"Pattern {idx}: {target}")