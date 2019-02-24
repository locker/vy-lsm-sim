Simulator of Tarantool Vinyl LSM tree implementation
====================================================

This simple script simulates performance of [Tarantool](http://github.com/tarantool/tarantool)
Vinyl LSM tree implementation in case of a write-only workload. It is useful for estimating read,
write, and space amplification for various Vinyl configurations.

Prerequisites
-------------

Python 3 and matplotlib are required to run the script.

Usage
-----

To run the script with the default parameters, run:
```
$ ./vy-lsm-sim.py
```

For the information about the script parameters, run:
```
$ ./vy-lsm-sim.py --help
```

Licensing
---------

The source code is free to use and redistribute under the terms of the FreeBSD License.
See [COPYING](https://github.com/locker/vy-lsm-sim/blob/master/COPYING) for the full license text.
