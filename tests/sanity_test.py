import multiprocessing
import pytest

# TODO: integrate with pytests testing suite

def sanity_test(algorithm, test_imgs):
    """ 
    Runs a sanity test on the algorithm that checks for run time and general exceptions.

    Args:
        algorithm: object that extends TaskPerceiver

    Example usage:
        ##### In Algorithm1.py #####
        class Algorithm1(TaskPerceiver):
            def analyze(self, frame, debug):
                pass

        ##### In test_Algorithm1.py #####
        from sanity_test import sanity_test
        import pytest

        # Some function that returns test images. This is scoped to this file/module.
        def get_test_imgs():
            return [None, None, None]

        @pytest.mark.parametrize("algorithm", [Algorithm1()])
        @pytest.mark.parametrize("test_imgs", [get_test_imgs()])
        def test_sanity(algorithm, test_imgs):
            sanity_test(algorithm, test_imgs)
    """

    MAX_RUNTIME = 3 # Per call to analyze() in seconds
    NUM_RUNS = 3 # Number of calls to analyze()

    if len(test_imgs) < NUM_RUNS:
        pytest.fail("Received less than {} test images".format(NUM_RUNS))

    # Run analyze() NUM_RUNS times and be ready to stop it if it takes too long
    pconn, cconn = multiprocessing.Pipe()
    p = multiprocessing.Process(target=lambda: run_algorithm(algorithm, test_imgs, NUM_RUNS, cconn))
    p.start()

    # Wait for MAX_RUNTIME * NUM_RUNS seconds or until the process finishes
    p.join(MAX_RUNTIME * NUM_RUNS)

    # If thread is still active, it took too long to finish
    if p.is_alive():
        p.terminate()
        p.join()
        pytest.fail("analyze() took over {} seconds with {} iterations."
                        .format(MAX_RUNTIME * NUM_RUNS, NUM_RUNS))
    # Check for exceptions
    if pconn.poll():
        error = pconn.recv()
        pytest.fail("analyze() encountered exception '{}' with test image {}".format(error[0], error[1]))

def run_algorithm(algorithm, test_imgs, num_runs, cconn):
    """ Wrapper function to run the algorithm on a separate thread. """
    for i in range(num_runs):
        try:
            algorithm.analyze(test_imgs[i], True)
        except Exception as e:
            # Send the error message and the image number back to the main thread
            cconn.send((e, i))
            break
            