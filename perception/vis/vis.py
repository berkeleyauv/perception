import argparse
import cProfile
import os
import sys

import cv2 as cv
import imageio
import numpy as np

from perception.tasks import registry
from perception.vis.FrameWrapper import FrameWrapper
from perception.vis.Visualizer import Visualizer


def _colorize(text, code, stream):
    # Skip escape codes when the stream isn't a terminal (piped/redirected
    # output, log files) - raw codes there would just show up as garbage.
    if not stream.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"


def _yellow(text):
    return _colorize(text, "33", sys.stdout)


def _red(text):
    return _colorize(text, "31", sys.stderr)


def _analyze(algorithm, window_builder, frame):
    if algorithm.kwargs:
        state, debug_frames = algorithm.analyze(frame, debug=True, slider_vals=window_builder.update_vars())
    else:
        state, debug_frames = algorithm.analyze(frame, debug=True)
    return state, window_builder.display(debug_frames)


def _label_frame(frame, text):
    # Burns a readable label into the top-left corner of a copy of frame -
    # used to tell the two halves of a --compare frame apart, since they
    # share one window instead of getting one each.
    labeled = frame.copy()
    font, scale, thickness = cv.FONT_HERSHEY_SIMPLEX, 0.6, 2
    (text_w, text_h), baseline = cv.getTextSize(text, font, scale, thickness)
    pad = 6
    cv.rectangle(labeled, (0, 0), (text_w + 2 * pad, text_h + baseline + 2 * pad), (0, 0, 0), -1)
    cv.putText(labeled, text, (pad, text_h + pad), font, scale, (0, 255, 0), thickness, cv.LINE_AA)
    return labeled


def _stack_compare(primary, compare):
    # Stack the compare grid below the primary one rather than beside it.
    # An algo's own debug grid is often already multi-column (2+ debug
    # frames), so putting two of those side-by-side doubles the window's
    # width - vstack instead keeps each pane's width (and thus its sub-frame
    # resolution and label/slider text size) unchanged.
    if primary.shape[1] != compare.shape[1]:
        target_w = primary.shape[1]
        scale = target_w / compare.shape[1]
        compare = cv.resize(compare, (target_w, int(compare.shape[0] * scale)))
    return np.vstack((primary, compare))


def run(data_sources, algorithm, save_video=False, resize=0.15, compare_algorithm=None,
        algo_label=None, compare_label=None, show_labels=True):
    out = None
    window_name = 'Debug Frames'
    compare_mode = compare_algorithm is not None
    # In compare mode both algos share one window, so their trackbars need
    # distinct names (label prefix) in case they use the same variable name.
    window_builder = Visualizer(algorithm.kwargs, window_name=window_name, label=algo_label if compare_mode else None)
    compare_window_builder = None
    if compare_mode:
        compare_window_builder = Visualizer(compare_algorithm.kwargs, window_name=window_name, label=compare_label)
    data = FrameWrapper(data_sources, resize)
    frame_count = 0
    speed = 1

    quit_requested = False
    for frame in data:
        if frame_count % speed == 0:
            _, to_show = _analyze(algorithm, window_builder, frame)
            if compare_mode:
                _, compare_to_show = _analyze(compare_algorithm, compare_window_builder, frame)
                if show_labels:
                    to_show = _label_frame(to_show, algo_label or 'primary')
                    compare_to_show = _label_frame(compare_to_show, compare_label or 'compare')
                to_show = _stack_compare(to_show, compare_to_show)
            cv.imshow(window_name, to_show)

            if save_video:
                if out is None:
                    out = imageio.get_writer('vis_rec.mp4')
                out.append_data(cv.cvtColor(to_show, cv.COLOR_BGR2RGB))
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


def profile(*args, stats='all', **kwargs):
    pr = cProfile.Profile()
    pr.enable()
    run(*args, **kwargs)
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
    parser.add_argument(
        '--algo',
        default=None,
        type=str,
        help='e.g. classical. If omitted, uses the task\'s default algo '
             '(its sole registered algo, or whichever was marked default=True).',
    )
    parser.add_argument(
        '--compare',
        default=None,
        type=str,
        help='Second algo for the same --task, e.g. classical. Runs it on the '
             'same frames and stacks it below the primary algo (each '
             'corner-labeled with its algo name) in one window, for direct '
             'comparison.',
    )
    parser.add_argument('--profile', default=None, type=str)
    parser.add_argument('--save_video', action='store_true')
    parser.add_argument(
        '--hide_labels',
        action='store_true',
        help='Hide the corner labels that identify each pane in --compare mode '
             '(shown by default).',
    )
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

    algo_name = args.algo
    if algo_name is None:
        try:
            algo_name = registry.get_default_algo(args.task)
        except KeyError as exc:
            raise SystemExit(_red(f"{exc}. Pass --algo explicitly.")) from None
        print(_yellow(f"No --algo given, using default for task {args.task!r}: {algo_name}"))

    try:
        algorithm = registry.get_perceiver(args.task, algo_name)()
    except KeyError as exc:
        available = ", ".join(
            f"{task}/{algo}"
            for task in registry.list_tasks()
            for algo in registry.list_algos(task)
        )
        raise SystemExit(_red(f"{exc}. Available: {available}")) from None

    compare_algorithm = None
    if args.compare is not None:
        try:
            compare_algorithm = registry.get_perceiver(args.task, args.compare)()
        except KeyError as exc:
            available = ", ".join(registry.list_algos(args.task))
            raise SystemExit(
                _red(f"{exc}. Available algos for task {args.task!r}: {available}")
            ) from None

    # Initialize image source
    # detects args.data, get a list of all file directory when given a directory
    # change data_source to a list of all files in the directory
    if os.path.isdir(args.data):
        data_sources = os.listdir(args.data)
    else:
        data_sources = [args.data]

    if args.profile is None:
        run(
            data_sources, algorithm, args.save_video, args.resize, compare_algorithm,
            algo_label=algo_name, compare_label=args.compare, show_labels=not args.hide_labels,
        )
    else:
        profile(
            data_sources, algorithm, args.save_video, args.resize, compare_algorithm,
            algo_label=algo_name, compare_label=args.compare, show_labels=not args.hide_labels,
            stats=args.profile,
        )
