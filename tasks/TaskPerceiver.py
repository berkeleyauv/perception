class TaskPerceiver:

    def __init__(self):
        self.time = 0

    def analyze(self, frame, debug):
        raise NotImplementedError("Need to implement with child class.")

    def calibrate(self, frame):
        raise NotImplementedError("Need to implement with child class.")

    def needs_calibration(self):
        return False

    # should be in Vis
    #def display(self, frame, debug):
    #    return self.analyze(frame, debug)
