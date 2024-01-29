"""
Functional HDF API
- operations get file state (hash) as an input, referring implicitely to the whole chain of previous operations
- write operation takes: file state, dataset, slice, data, returns new file state
- read takes: file state, dataset, slice; returns pair (file state, numpy array)

This does not allow skipping completed operations itself, need storing completed hashes retrieving completed states.

Temporary solution:
- open file just ofr single operations, catch filure, retry in some wait time, seems that HDF does not provide such functionality


It is quite difficult to deal with unreliable NFS in particular how to deal with failed file operation.
We may end up with broken file, without ability of HDF to recover from such state
"""

def read()