import argparse
import os

from perception import get_perceiver
from perception.tasks.TaskPerceiver import TaskContext
from perception.vis.FrameWrapper import FrameWrapper
import cv2 as cv
from perception.vis.Visualizer import Visualizer
import cProfile


LEGACY_ALIASES = {
    "test": ("examples", "test"),
    "gateseg": ("gate", "classical"),
    "gatesegA": ("gate", "segmentation_a"),
    "gatesegB": ("gate", "segmentation_b"),
    "gatesegC": ("gate", "segmentation_c"),
}


def run(data_sources, algorithm, save_video=False, compare_algorithm=None):
    out = None
    imageio = None
    if save_video:
        import imageio

    window_builder = Visualizer(algorithm.kwargs)
    data = FrameWrapper(data_sources, 0.15)
    frame_count = 0
    speed = 1

    for frame in data:
        if frame_count % speed == 0:
            output = algorithm.predict(
                frame,
                context=TaskContext(
                    debug=True,
                    tunables=window_builder.update_vars(),
                    frame_id=str(frame_count),
                ),
            )
            debug_frames = list(output.debug_frames.values()) or [frame]
            if compare_algorithm:
                compare_output = compare_algorithm.predict(
                    frame,
                    context=TaskContext(
                        debug=True,
                        frame_id=str(frame_count),
                    ),
                )
                debug_frames.extend(list(compare_output.debug_frames.values()) or [frame])

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


def build_algorithm(task, algo, legacy_algorithm=None):
    if legacy_algorithm:
        task, algo = LEGACY_ALIASES.get(legacy_algorithm, (None, None))
        if task is None:
            raise KeyError(f"Unknown legacy algorithm alias {legacy_algorithm}")
    if not task or not algo:
        raise ValueError("Provide --task and --algo, or a legacy --algorithm alias.")
    return get_perceiver(task, algo)()


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Visualizes perception algorithms.')
    parser.add_argument('--data', default='webcam', type=str)
    parser.add_argument('--task', type=str)
    parser.add_argument('--algo', type=str)
    parser.add_argument('--compare', type=str)
    parser.add_argument('--algorithm', type=str)
    parser.add_argument('--profile', default=None, type=str)
    parser.add_argument('--save_video', action='store_true')
    args = parser.parse_args()

    # Get algorithm class and init
    algorithm = build_algorithm(args.task, args.algo, args.algorithm)
    compare_algorithm = None
    if args.compare:
        compare_algorithm = build_algorithm(args.task, args.compare)

    # Initialize image source
    # detects args.data, get a list of all file directory when given a directory
    # change data_source to a list of all files in the directory
    if os.path.isdir(args.data):
        data_sources = os.listdir(args.data)
    else:
        data_sources = [args.data]

    if args.profile is None:
        run(data_sources, algorithm, args.save_video, compare_algorithm=compare_algorithm)
    else:
        profile(data_sources, algorithm, args.save_video, compare_algorithm, stats=args.profile)
