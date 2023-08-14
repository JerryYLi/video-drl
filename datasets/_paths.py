import os.path

K400_DATA = [
    '/data/datasets/kinetics/video/',
]

K400_META = [
    '/data/datasets/kinetics/anno/kinetics400/',
]

K200_META = [
    '/data/datasets/kinetics/anno/minikinetics200/',
]

UCF_DATA = [
    '/data/datasets/ucf101/data/',
]

UCF_META = [
    '/data/datasets/ucf101/ucfTrainTestlist/',
]

HMDB_DATA = [
    '/data/datasets/HMDB/videos/',
]

HMDB_META = [
    '/data/datasets/HMDB/testTrainMulti_7030_splits/',
]

STHV1_DATA = [
    '/data/datasets/something/v1/20bn-something-something-v1/',
]

STHV1_META = [
    '/data/datasets/something/v1/anno/',
]

STHV2_DATA = [
    '/data/datasets/something/v2/20bn-something-something-v2/',
]

STHV2_META = [
    '/data/datasets/something/v2/anno/',
]

JESTER_DATA = [
    '/data/datasets/jester/20bn-jester-v1/',
]

JESTER_META = [
    '/data/datasets/jester/anno/',
]

DIVING_DATA = [
    '/data/datasets/diving/rgb/',
]

DIVING_META = [
    '/data/datasets/diving/anno/',
]


def find_path(paths):
    for path in paths:
        if os.path.exists(path):
            return path