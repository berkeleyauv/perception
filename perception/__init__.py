import perception.vis.TestAlgo as TestAlgo
import perception.tasks.gate.GateCenterAlgo as GateSeg
import perception.tasks.gate.GateSegmentationAlgoA as GateSegA
import perception.tasks.gate.GateSegmentationAlgoB as GateSegB
import perception.tasks.gate.GateSegmentationAlgoC as GateSegC
import perception.tasks.segmentation.saliency_detection.MBD as MBD
# import perception.tasks as tasks

ALGOS = {
    'test': TestAlgo.TestAlgo,
    'gateseg': GateSeg.GateCenterAlgo,
    'gatesegA': GateSegA.GateSegmentationAlgoA,
    'gatesegB': GateSegB.GateSegmentationAlgoB,
    'gatesegC': GateSegC.GateSegmentationAlgoC,
    'MBD': MBD.MBD
}
