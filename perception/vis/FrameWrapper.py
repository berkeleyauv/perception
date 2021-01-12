import cv2
import sys

class FrameWrapper():
    """
    A standard interface for getting frames from images, videos, and the webcam.
    TODO: ZED camera interfacing

    Example usage:
        filenames = ['relative/path/img.png', './video.mp4', 'webcam']
        frames = FrameWrapper(filenames)

        # Shows img.png, then all frames in video.mp4, then frames from webcam forever
        for frame in frames:
            cv2.imshow('Next frame', frame)
    """

    # Keywords used to identify file types
    WEBCAM = ['webcam']
    VIDEO_EXTS = ['mp4', 'avi']
    IMG_EXTS = ['jpg', 'png']

    VIDEO_TRIES = 200
    WEBCAM_TRIES = 10

    def __init__(self, filenames, resize=1):
        self.filenames = filenames  # Get this list of relative paths to files from vis
        # There aren't any checks for resize==1 to improve speed b/c this expects resize != 1
        self.resize = resize

    def __iter__(self):
        self.index = -1
        self.next_data = ('', None)
        self.next_data_obj()
        return self

    def __next__(self):
        if not self.has_next:
            raise StopIteration

        while self.index < len(self.filenames):
            if self.next_data[0] == "v": # Video
                # Try to get a frame out at most VIDEO_TRIES times.
                # If it still fails, we're probably at the end of the video file
                for i in range(self.VIDEO_TRIES):
                    ret, frame = self.next_data[1].read()
                    if ret:
                        return cv2.resize(frame, None, fx=self.resize, fy=self.resize)
                    print("WARNING: Failed to get frame from video {}. Try {}." \
                            .format(self.filenames[self.index], i), file=sys.stderr)
                self.next_data_obj()
            elif self.next_data[0] == "i": # Image
                img = self.next_data[1]
                self.next_data_obj()
                if img is not None:
                    return cv2.resize(img, None, fx=self.resize, fy=self.resize)
                print("WARNING: Failed to get image {}." \
                        .format(self.filenames[self.index-1]), file=sys.stderr)
            else: # Webcam
                # Try to get a frame out at most WEBCAM_TRIES times.
                for i in range(self.WEBCAM_TRIES):
                    ret, frame = self.next_data[1].read()
                    if ret:
                        return cv2.resize(frame, None, fx=self.resize, fy=self.resize)
                    print("WARNING: Failed to get frame from webcam. Try {}." \
                            .format(i), file=sys.stderr)
                self.next_data_obj()

        raise StopIteration

    def next_data_obj(self):
        """
        Helper function for getting the next object (video, image, webcam) when
        the previous one is exhausted.
        """
        # Close the webcam if it was open and we don't want it anymore
        if self.next_data[0] in self.WEBCAM:
            self.next_data[1].release()

        # Stop if we don't have any more data
        if self.index >= len(self.filenames) - 1:
            self.index += 1
            self.has_next = False
            return

        # Prepare the next data object
        self.index += 1
        filename = self.filenames[self.index]
        extension = filename[filename.rindex('.') + 1:].lower() if '.' in filename else None

        if filename in self.WEBCAM:
            self.next_data = ('w', cv2.VideoCapture(0))
        elif extension in self.VIDEO_EXTS:
            self.next_data = ('v', cv2.VideoCapture(filename))
        elif extension in self.IMG_EXTS:
            self.next_data = ('i', cv2.imread(filename))
        else:
            print("Unknown file format:", extension)

        self.has_next = True
