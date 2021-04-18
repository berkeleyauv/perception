import perception.vis.TestAlgo as TestAlgo
import perception.tasks.gate.GateCenterAlgo as GateSeg
import perception.tasks.gate.GateSegmentationAlgoA as GateSegA
import perception.tasks.gate.GateSegmentationAlgoB as GateSegB
import perception.tasks.gate.GateSegmentationAlgoC as GateSegC
import perception.tasks.segmentation.saliency_detection.MBD as MBD
from perception.tasks.segmentation.COMB_SAL_BG import COMB_SAL_BG
import perception.vis.TestTasks.BackgroundRemoval as BackgroundRemoval
import perception.tasks.roulette.color_detection as RouletteColorDetector
from perception.tasks.dice.DiceDetector import DiceDetector

ALGOS = {
    'test': TestAlgo.TestAlgo,
    'gateseg': GateSeg.GateCenterAlgo,
    'gatesegA': GateSegA.GateSegmentationAlgoA,
    'gatesegB': GateSegB.GateSegmentationAlgoB,
    'gatesegC': GateSegC.GateSegmentationAlgoC,
    'MBD': MBD.MBD,
    'bg-rm': BackgroundRemoval.BackgroundRemoval,
    'combined': COMB_SAL_BG,
    'roulette': RouletteColorDetector.RouletteColorDetector,
    'dice': DiceDetector
}
