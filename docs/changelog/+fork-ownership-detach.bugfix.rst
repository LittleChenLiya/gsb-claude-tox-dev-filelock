A lock object that detects it was inherited across ``fork()`` now explicitly detaches the inherited hold: it
closes only its own inherited descriptor copies and drops the parent process's reentrant counter, without
calling the backend unlock or unlink that would tear down the parent process's lock. Previously ``release()``
and garbage collection in the child returned silently, leaking the inherited descriptor copies.
