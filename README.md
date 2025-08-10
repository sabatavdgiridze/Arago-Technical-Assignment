To run the Python script, we must first install the libraries:
```bash
pip install torch
```
Then we run it
```bash
python Optimizer.py
```

The script implements the algorithm described in the solution (for question 7 in answer's document). For both the uniform density (sanity check) and also the linear density (p(x) = 2x) with different initial values for the triplet (a, b, c). I used Adam Optimizer for this problem. Also the evaluation points of the integral and constructed at the beginning by sampling points from the distribution. This in principle should help it be more dense where it matters (the pdf is highest). The number of grid points were 1,000,000 by default, and the iteration count 10,000. I chose the learning rate to be 0.02 because this seemed to give the best results. Until Epoch 1,000 the loss was decreasing, afterwards it started oscillating along with parameters (a, b, c). 
