import argparse
from perception.vis.FrameWrapper import FrameWrapper
import cv2 as cv
from perception.vis.Visualizer import Visualizer
import cProfile


def run(data_sources, algorithm, save_video=False):
    out = None
    window_builder = Visualizer(algorithm.kwargs)
    data = FrameWrapper(data_sources, 0.25)
    frame_count = 0
    paused = False
    speed = 1

    for frame in data:
        if frame_count % speed == 0 and not paused:
            if algorithm.kwargs:
                state, debug_frames = algorithm.analyze(frame, debug=True, slider_vals=window_builder.update_vars())
            else:
                state, debug_frames = algorithm.analyze(frame, debug=True)

            to_show = window_builder.display(debug_frames)
            cv.imshow('Debug Frames', to_show)
            if save_video:
                if out is None:
                    height, width, _ = to_show.shape
                    # TODO: get codec to work
                    out = cv.VideoWriter('rec.mp4', cv.VideoWriter_fourcc(*'mp4v'), 60, (height, width))
                if out:
                    out.write(to_show)

        key = cv.waitKey(30)
        if key == ord('q') or key == 27:
            break
        if key == ord('p'):
            paused = not paused
        if key == ord('i') and speed > 1:
            speed -= 1
            print(f'speed {speed}')
        if key == ord('o'):
            speed += 1
            print(f'speed {speed}')
        frame_count += 1

    cv.destroyAllWindows()
    if out:
        out.release()


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Visualizes perception algorithms.')
    parser.add_argument('--data', default='webcam', type=str)
    parser.add_argument('--algorithm', type=str)
    parser.add_argument('--save_video', action='store_true')
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()

    # Import Algorithm
    exec(f"from {args.algorithm} import {args.algorithm.split('.')[-1]} as Algorithm")
    algorithm = Algorithm()
    data_sources = [args.data]

    if args.profile:
        stats = cProfile.run('run(data_sources, algorithm, args.save_video)')
    else:
        run(data_sources, algorithm, args.save_video)
