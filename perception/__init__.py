import perception.vis.TestAlgo as TestAlgo
import perception.tasks.gate.classical.GateCenterAlgo as GateSeg
import perception.tasks.gate.classical.GateSegmentationAlgoA as GateSegA
import perception.tasks.gate.classical.GateSegmentationAlgoB as GateSegB
import perception.tasks.gate.classical.GateSegmentationAlgoC as GateSegC
import perception.vis.TestTasks.BackgroundRemoval as BackgroundRemoval

try:
    import perception.tasks.segmentation.saliency_detection.MBD as MBD
except ModuleNotFoundError:
    MBD = None

try:
    from perception.tasks.segmentation.COMB_SAL_BG import COMB_SAL_BG
except ModuleNotFoundError:
    COMB_SAL_BG = None

try:
    import perception.tasks._archive.roulette.color_detection as RouletteColorDetector
except ModuleNotFoundError:
    RouletteColorDetector = None

try:
    from perception.tasks._archive.dice.DiceDetector import DiceDetector
except ModuleNotFoundError:
    DiceDetector = None

ALGOS = {
    'test': TestAlgo.TestAlgo,
    'gateseg': GateSeg.GateCenterAlgo,
    'gatesegA': GateSegA.GateSegmentationAlgoA,
    'gatesegB': GateSegB.GateSegmentationAlgoB,
    'gatesegC': GateSegC.GateSegmentationAlgoC,
    'bg-rm': BackgroundRemoval.BackgroundRemoval,
}

if MBD is not None:
    ALGOS['MBD'] = MBD.MBD

if COMB_SAL_BG is not None:
    ALGOS['combined'] = COMB_SAL_BG

if RouletteColorDetector is not None:
    ALGOS['roulette'] = RouletteColorDetector.RouletteColorDetector

if DiceDetector is not None:
    ALGOS['dice'] = DiceDetector
