__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Jun 20, 2014 12:07:48 EDT$"


#import nanshe.advanced_debugging


#logger = nanshe.advanced_debugging.logging.getLogger(__name__)




class SPAMSException(Exception):
    pass


#@nanshe.advanced_debugging.log_call(logger)
def run_multiprocessing_queue_spams_trainDL(out_queue, *args, **kwargs):
    """
        Designed to run spams.trainDL in a separate process.

        It is necessary to run SPAMS in a separate process as segmentation faults
        have been discovered in later parts of the Python code dependent on whether
        SPAMS has run or not. It is suspected that spams may interfere with the
        interpreter. Thus, it should be sandboxed (run in a different Python interpreter)
        so that it doesn't damage what happens in this one.

        This particular version uses a multiprocessing.Queue to return the resulting dictionary.


        Args:
            out_queue(multiprocessing.Queue):       what will take the returned dictionary from spams.trainDL.
            *args(list):                            a list of position arguments to pass to spams.trainDL.
            *kwargs(dict):                          a dictionary of keyword arguments to pass to spams.trainDL.
    """

    # It is not needed outside of calling spams.trainDL.
    # Also, it takes a long time to load this module.
    import spams

    result = spams.trainDL(*args, **kwargs)
    out_queue.put(result)


#@nanshe.advanced_debugging.log_call(logger)
def call_multiprocessing_queue_spams_trainDL(*args, **kwargs):
    """
        Designed to start spams.trainDL in a separate process and handle the result in an unnoticeably different way.

        It is necessary to run SPAMS in a separate process as segmentation faults
        have been discovered in later parts of the Python code dependent on whether
        SPAMS has run or not. It is suspected that spams may interfere with the
        interpreter. Thus, it should be sandboxed (run in a different Python interpreter)
        so that it doesn't damage what happens in this one.

        This particular version uses a multiprocessing.Queue to return the resulting dictionary.


        Args:
            *args(list):                            a list of position arguments to pass to spams.trainDL.
            *kwargs(dict):                          a dictionary of keyword arguments to pass to spams.trainDL.

        Returns:
            result(numpy.matrix): the dictionary found
    """

    # Only necessary for dealing with SPAMS
    import multiprocessing

    out_queue = multiprocessing.Queue()

    p = multiprocessing.Process(target = run_multiprocessing_queue_spams_trainDL, args = (out_queue,) + args, kwargs = kwargs)
    p.start()
    result = out_queue.get()
    result = result.copy()
    p.join()

    if p.exitcode != 0: raise SPAMSException("SPAMS has terminated with exitcode \"" + repr(p.exitcode) + "\".")

    return(result)


#@nanshe.advanced_debugging.log_call(logger)
def run_multiprocessing_array_spams_trainDL(result_array_type, result_array, X_array_type, X_array, *args, **kwargs):
    """
        Designed to start spams.trainDL in a separate process and handle the result in an unnoticeably different way.

        It is necessary to run SPAMS in a separate process as segmentation faults
        have been discovered in later parts of the Python code dependent on whether
        SPAMS has run or not. It is suspected that spams may interfere with the
        interpreter. Thus, it should be sandboxed (run in a different Python interpreter)
        so that it doesn't damage what happens in this one.

        This particular version uses a multiprocessing.Array to share memory to return the resulting dictionary.


        Args:
            result_array_type(numpy.ctypeslib.ndpointer):   a pointer type with properties needed by result_array.
            result_array(multiprocessing.RawArray):         shared memory array to store results in.
            X_array_type(numpy.ctypeslib.ndpointer):        a pointer type with properties needed by X_array.
            X_array(numpy.ndarray):                         currently uses numpy ndarray as input.
            *args(list):                                    a list of position arguments to pass to spams.trainDL.
            *kwargs(dict):                                  a dictionary of keyword arguments to pass to spams.trainDL.

        Note:
            This is somewhat faster than using multiprocessing.Queue.
    """

    # Just to make sure this exists in the new process. Shouldn't be necessary.
    import numpy
    # Just to make sure this exists in the new process. Shouldn't be necessary.
    # Also, it is not needed outside of calling this function.
    import spams

    as_ordered_array_dict = {
        "F_CONTIGUOUS" : numpy.asfortranarray,
        "C_CONTIGUOUS" : numpy.ascontiguousarray
    }


    # Construct X from shared array.
    X_dtype = X_array_type._dtype_
    X_shape = X_array_type._shape_
    X_flags = numpy.core.multiarray.flagsobj(X_array_type._flags_)

    X = numpy.frombuffer(X_array, dtype = X_dtype).reshape(X_shape)
    X.setflags(X_flags)

    for order_name, as_ordered_array in as_ordered_array_dict.items():
        if order_name in X_array_type.__name__:
            X = as_ordered_array(X)

    # Construct the result to use the shared buffer.
    result_dtype = result_array_type._dtype_
    result_shape = result_array_type._shape_
    result_flags = numpy.core.multiarray.flagsobj(result_array_type._flags_)

    result = numpy.frombuffer(result_array, dtype = result_dtype).reshape(result_shape)
    result.setflags(result_flags)

    for order_name, as_ordered_array in as_ordered_array_dict.items():
        if order_name in result_array_type.__name__:
            result = as_ordered_array(result)


    result[:] = spams.trainDL(X, *args, **kwargs)


#@nanshe.advanced_debugging.log_call(logger)
def call_multiprocessing_array_spams_trainDL(X, *args, **kwargs):
    """
        Designed to start spams.trainDL in a separate process and handle the result in an unnoticeably different way.

        It is necessary to run SPAMS in a separate process as segmentation faults
        have been discovered in later parts of the Python code dependent on whether
        SPAMS has run or not. It is suspected that spams may interfere with the
        interpreter. Thus, it should be sandboxed (run in a different Python interpreter)
        so that it doesn't damage what happens in this one.

        This particular version uses a multiprocessing.Array to share memory to return the resulting dictionary.


        Args:
            X(numpy.matrix):                        a Fortran order NumPy Matrix with the same name as used by spams.trainDL (so if someone tries to use it as a keyword argument...).
            *args(list):                            a list of position arguments to pass to spams.trainDL.
            **kwargs(dict):                         a dictionary of keyword arguments to pass to spams.trainDL.

        Note:
            This is somewhat faster than using multiprocessing.Queue.
    """

    # Only necessary for dealing with SPAMS
    import multiprocessing

    # Just to make sure this exists in the new process. Shouldn't be necessary.
    import numpy


    # Types for X_array
    X_array_type = numpy.ctypeslib.ndpointer(dtype=X.dtype, ndim=X.ndim, shape=X.shape, flags=X.flags)
    X_array_ctype = type(numpy.ctypeslib.as_ctypes(X_array_type._dtype_.type(0)[()]))

    # Create a shared array to contain X
    X_array = multiprocessing.Array(X_array_ctype,
                                    X.size,
                                    lock=False)

    # Copy over the contents of X.
    X_array_numpy = numpy.frombuffer(X_array, dtype=X_array_type._dtype_).reshape(X_array_type._shape_)
    X_array_numpy[:] = X
    X_array_numpy = None


    # Types for result_array
    result_array_type = numpy.ctypeslib.ndpointer(dtype=X.dtype, ndim=X.ndim, shape=(X.shape[0], kwargs["K"]))
    result_array_ctype = type(numpy.ctypeslib.as_ctypes(result_array_type._dtype_.type(0)[()]))

    # Create a shared array to contain the result
    result_array = multiprocessing.Array(result_array_ctype,
                                         numpy.product(result_array_type._shape_),
                                         lock=False)


    p = multiprocessing.Process(target = run_multiprocessing_array_spams_trainDL, args = (result_array_type, result_array, X_array_type, X_array,) + args, kwargs = kwargs)
    p.start()
    p.join()

    if p.exitcode != 0: raise SPAMSException("SPAMS has terminated with exitcode \"" + repr(p.exitcode) + "\".")


    # Reconstruct the result from the output array
    result = numpy.frombuffer(result_array, dtype = result_array_type._dtype_).reshape(result_array_type._shape_)
    result = result.copy()

    return(result)


#@nanshe.advanced_debugging.log_call(logger)
def call_spams_trainDL(*args, **kwargs):
    """
        Encapsulates call to spams.trainDL. Ensures copy of results occur just in case.
        Designed to be like the multiprocessing calls.

        Args:
            *args(list):                            a list of position arguments to pass to spams.trainDL.
            **kwargs(dict):                         a dictionary of keyword arguments to pass to spams.trainDL.

        Note:
            For legacy.
    """

    # It is not needed outside of calling spams.trainDL.
    # Also, it takes a long time to load this module.
    import spams

    result = spams.trainDL(*args, **kwargs)
    result = result.copy()

    return(result)
