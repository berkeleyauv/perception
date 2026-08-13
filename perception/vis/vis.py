import argparse
import cProfile
import os

import cv2 as cv
import imageio

from perception.tasks import registry
from perception.vis.FrameWrapper import FrameWrapper
from perception.vis.Visualizer import Visualizer


def run(data_sources, algorithm, save_video=False, resize=0.15):
    out = None
    window_builder = Visualizer(algorithm.kwargs)
    data = FrameWrapper(data_sources, resize)
    frame_count = 0
    speed = 1

    quit_requested = False
    for frame in data:
        if frame_count % speed == 0:
            if algorithm.kwargs:
                state, debug_frames = algorithm.analyze(frame, debug=True, slider_vals=window_builder.update_vars())
            else:
                state, debug_frames = algorithm.analyze(frame, debug=True)

            to_show = window_builder.display(debug_frames)
            cv.imshow('Debug Frames', to_show)
            if save_video:
                if out is None:
                    out = imageio.get_writer('vis_rec.mp4')
                out_img = cv.cvtColor(to_show, cv.COLOR_BGR2RGB)
                out.append_data(out_img)
        frame_count += 1

        key = cv.waitKey(30)
        if key == ord('q') or key == 27:
            quit_requested = True
            break
        if key == ord('p'):
            cv.waitKey(0)  # pause
            # TODO: be able to quit and manipulate slider vars in real time while paused
        if key == ord('i') and speed > 1:
            speed -= 1
            print(f'speed {speed}')
        if key == ord('o'):
            speed += 1
            print(f'speed {speed}')


    if frame_count > 0 and not quit_requested:
        # Hold the last frame open until a key is pressed instead of tearing the
        # window down instantly - matters for single-image input, which would
        # otherwise flash on screen for one 30ms waitKey and vanish. Skipped if
        # the user already pressed q/Esc to quit, since that's an explicit exit.
        # Poll in short bursts rather than a single blocking waitKey(0): a fully
        # blocking waitKey() doesn't return control to Python until a key is
        # pressed in the window, which also blocks Ctrl-C (SIGINT) from being
        # handled until then.
        while cv.waitKey(30) == -1:
            pass
    cv.destroyAllWindows()
    if out:
        out.close()


def profile(*args, stats='all'):
    pr = cProfile.Profile()
    pr.enable()
    run(*args)
    pr.disable()
    if stats == 'all':
        pr.print_stats()
    else:
        pr.print_stats(stats)


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Visualizes perception algorithms.')
    parser.add_argument('--data', default='webcam', type=str)
    parser.add_argument(
        '--task', type=str, required=True, help='e.g. slalom, gate, path_marker'
    )
    parser.add_argument('--algo', type=str, required=True, help='e.g. classical')
    parser.add_argument('--profile', default=None, type=str)
    parser.add_argument('--save_video', action='store_true')
    parser.add_argument(
        "--resize",
        default=1.0,
        type=float,
        help="Scale factor applied to every frame before display (default: 1.0, no resize).",
    )
    args = parser.parse_args()

    # Discover every @register_perceiver in perception.tasks, then look up the
    # requested one. No shared file needs hand-editing to add a new algorithm.
    registry.discover_all()
    try:
        algorithm = registry.get_perceiver(args.task, args.algo)()
    except KeyError as exc:
        available = ", ".join(
            f"{task}/{algo}"
            for task in registry.list_tasks()
            for algo in registry.list_algos(task)
        )
        raise SystemExit(f"{exc}. Available: {available}") from None

    # Initialize image source
    # detects args.data, get a list of all file directory when given a directory
    # change data_source to a list of all files in the directory
    if os.path.isdir(args.data):
        data_sources = os.listdir(args.data)
    else:
        data_sources = [args.data]

    if args.profile is None:
        run(data_sources, algorithm, args.save_video, args.resize)
    else:
        profile(
            data_sources, algorithm, args.save_video, args.resize, stats=args.profile
        )
