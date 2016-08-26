# kombine

**A kernel-density-based, embarrassingly parallel ensemble sampler**

[![Travis CI build status (Linux)](https://travis-ci.org/bfarr/kombine.svg?branch=master)](https://travis-ci.org/bfarr/kombine)
[![Code Coverage](https://codecov.io/gh/bfarr/kombine/branch/master/graph/badge.svg)](https://codecov.io/gh/bfarr/kombine)

kombine is an ensemble sampler that uses a clustered
kernel-density-estimate proposal density, allowing for *massive*
parallelization and efficient sampling.

## Documentation

Example usage:

 * [2-D Gaussian](http://nbviewer.ipython.org/github/bfarr/kombine/blob/master/examples/2D_gaussian.ipynb)
 * [2-D Rosenbrock](http://nbviewer.ipython.org/github/bfarr/kombine/blob/master/examples/rosenbrock.ipynb)


## Attribution
```
@article{kombine,
   author = {{Farr}, B. and {Farr}, W.~M.},
    title = {kombine: a kernel-density-based, embarrassingly parallel ensemble sampler},
     year = 2015,
     note = "in prep"
}
```

## License

Copyright 2014 Ben Farr and contributors.

kombine is free software made available under the MIT License. For details see the LICENSE file.
