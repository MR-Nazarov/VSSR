from .dataset import *
from monai.data import Dataset as MonaiDataset
from monai.data import CacheDataset as CachedMonaiDataset
from monai.data import load_decathlon_datalist


class _testDataset(datasetClass):
    def __init__(self, args):
        super().__init__(args)
        self.datalist = load_decathlon_datalist(data_list_file_path=args.jsonData,
                                                is_segmentation=self.args.segmentation,
                                                data_list_key=self.args.keyTest)


class _trainDataset(datasetClass):
    def __init__(self, args):
        super().__init__(args)
        key = 'training' if \
            self.args.trainPhase \
            else 'validation'
        self.datalist = load_decathlon_datalist(data_list_file_path=args.jsonData,
                                                is_segmentation=self.args.segmentation,
                                                data_list_key=key)


class testDataset(MonaiDataset, _testDataset):

    def __init__(self, args):
        self.dataIdx = []
        _testDataset.__init__(self, args)
        transforms = getattr(self, 'valTransforms',
                             getattr(self, 'transforms'))
        super(testDataset, self).__init__(data=self.datalist,
                                          transform=transforms)

    def __getitem__(self, idx):
        self.dataIdx.append(idx)
        return super(testDataset, self).__getitem__(idx)

    def read_data(self):
        """
        Only to override the read_data of the base classes
        """

    def post_set_data(self):
        """
        Only to override the read_data of the base classes
        """

    def post_read_data(self, data):
        """
        Only to override the read_data of the base classes
        """


class trainDataset(MonaiDataset, _trainDataset):

    def __init__(self, args):
        # create transforms and read json file using decathlon - monai.data.load_decathlon_datalist()
        _trainDataset.__init__(self, args)
        transformName = 'transforms'
        if not self.args.trainPhase:
            transformName = 'valTransforms'
        transforms = getattr(self, transformName, getattr(self, 'transforms'))
        super(trainDataset, self).__init__(data=self.datalist,
                                           transform=transforms)

    def read_data(self):
        """
        Only to override the read_data of the base classes
        """

    def post_set_data(self):
        """
        Only to override the read_data of the base classes
        """

    def post_read_data(self, data):
        """
        Only to override the read_data of the base classes
        """


class testDatasetCached(CachedMonaiDataset, _testDataset):

    def __init__(self, args):
        _testDataset.__init__(self, args)
        transforms = getattr(self, 'valTransforms',
                             getattr(self, 'transforms'))
        super(testDatasetCached, self).__init__(data=self.datalist,
                                                transform=transforms,
                                                cache_num=self.args.cache_num,
                                                cache_rate=self.args.cache_rate,
                                                num_workers=args.workers)


class trainDatasetCached(CachedMonaiDataset, _trainDataset):

    def __init__(self, args):
        _trainDataset.__init__(self, args)
        transformName = 'transforms'
        if not self.args.trainPhase:
            transformName = 'valTransforms'
        transforms = getattr(self, transformName, getattr(self, 'transforms'))
        super(trainDatasetCached, self).__init__(data=self.datalist,
                                                 transform=transforms,
                                                 cache_num=self.args.cache_num,
                                                 cache_rate=self.args.cache_rate,
                                                 num_workers=args.workers)

    def read_data(self):
        """
        Only to override the read_data of the base classes
        """

    def post_set_data(self):
        """
        Only to override the read_data of the base classes
        """

    def post_read_data(self, data):
        """
        Only to override the read_data of the base classes
        """
