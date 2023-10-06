---
title: "Multiprocessing errors in PyTorch"
date: 2023-05-25T16:46:13-01:00
draft: false
format: hugo-md
jupyter: python3
---
When you get your own OS-specific errors you have to start wondering whether the mere act of programming on Windows is a mistake.

```
An attempt has been made to start a new process before the
current process has finished its bootstrapping phase.

This probably means that you are not using fork to start your
child processes and you have forgotten to use the proper idiom
in the main module:

        if __name__ == '__main__':
        freeze_support()
        ...

The "freeze_support()" line can be omitted if the program
is not going to be frozen to produce an executable.
```
 This error snippet comes from Pytorch's implementation of multiprocessing and can show up during execution even when your code doesn't explicitly do any--simply specifying a paramter such as `num_workers` is an implicit use of the feature and invites its caveats. The warning refers to the use of `fork` and starting subprocesses, but the choice of using it is not explicit either and simply is a consequence of what OS your machine is running.

According to the PyTorch docs:
> Since workers rely on Python multiprocessing, worker launch behavior is different on Windows compared to Unix.
> - On Unix, fork() is the default multiprocessing start method. Using fork(), child workers typically can access the dataset and Python argument functions directly through the cloned address space.
> - On Windows or MacOS, spawn() is the default multiprocessing start method. Using spawn(), another interpreter is launched which runs your main script, followed by the internal worker function that receives the dataset, collate_fn and other arguments through pickle serialization.

Unix users won't see this specific error because its implementation allows direct access to data and variables. Elsewhere, the `spawn` implementation causes independent processes to start up and when that happens, all the code will be run a second (or nth) time unless precautions are taken:
> This separate serialization means that you should take two steps to ensure you are compatible with Windows while using multi-process data loading:
> - Wrap most of you main script’s code within if __name__ == '__main__': block, to make sure it doesn’t run again (most likely generating error) when each worker process is launched. You can place your dataset and DataLoader instance creation logic here, as it doesn’t need to be re-executed in workers.
> - Make sure that any custom collate_fn, worker_init_fn or dataset code is declared as top level definitions, outside of the __main__ check. This ensures that they are available in worker processes. (this is needed since functions are pickled as references only, not bytecode.)

![Listen to Boromir](https://i.imgflip.com/7o7o4c.jpg)

If the `if __name__ == '__main__'` guard idiom referenced seems novel, then your scripts have mostly been self-contained until now. Many if not all Python libraries make use of this to provide executable script functionality while still being importable as modules. The __name__ variable is a Python default that's automatically set to '__main__' for the script that is directly executed. The idiom simply takes advantage of that fact to create a conditional, under which you place the separated code that's intended to execute only when the script is run directly.

In the PyTorch example, code inside the guard won't be run again as the multiprocessing workers launch, but it's also correct to say that the same code won't run upon being imported to another script. Anything outside the guard, such as a `df` or other global variable, will execute during import to other scripts.