This repository contains the library `equichain` that was used to perform
the computations in the paper:

https://arxiv.org/abs/1907.08204

# Installation

The prerequisite is Sage, which can be downloaded at:

https://www.sagemath.org/

You will have to compile it from source. (The binary packages available on
SageMath don't let you install optional Sage packages, which you will need.
Although there are also now Sage packages available in the Ubuntu repositories,
I wasn't able to successfully run `equichain` using these packages, at least on
Ubuntu 18.04).

Performance will be greatly improved if you use the patched Sage sources
available at [https://github.com/dominicelse/sage/], which implement some more
efficient routines for initializing Sage matrices that this code can take
advantage of.

As Sage doesn't provide routines for sparse matrix linear algebra over the
integers or finite fields, `equichain` uses the dense matrix routines instead.
There is an option to use the sparse matrix routines from Magma, which will
greatly improve performance and memory usage. The Magma website can be found at
[http://magma.maths.usyd.edu.au/magma/], but the software is not open source and
requires a pretty hefty license fee unless your institution already has a site
license. If Magma is installed properly, i.e. the `magma` command is in your
`PATH`, then `equichain` will use it automatically. (A warning will be spit out
if Magma is not available.)


# Usage

After you've installed Sage, you just need to ensure that the `equichain/`
subdirectory is referenced in the `PYTHONPATH` environment variable. Then you can
import the `equichain` module, for instance. Note that before running any
intensive computations, you may want to run

```
import sage.interfaces.gap
sage.interfaces.gap.gap = sage.interfaces.gap.Gap(max_workspace_size = "4g")
```
(replace `"4g"` with an appropriate value) to increase the amount of memory available to GAP.

The main workhorse for the computations in the paper is the
`equichain.lsm.check_space_group()` function. Type
```
import equichain.lsm
help(equichain.lsm.check_space_group)
```
in the Sage interpreter for more information about this function.
