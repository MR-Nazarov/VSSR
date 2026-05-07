from datasets.monai_dataset import testDataset as testDatasetMonai
from datasets.monai_dataset import trainDataset as trainDatasetMonai
from datasets.monai_dataset import testDatasetCached as testDatasetMonaiCached
from datasets.monai_dataset import trainDatasetCached as trainDatasetMonaiCached
from datasets.denoising_dataset import testDataset as testDatasetDenoising
from datasets.denoising_dataset import trainDataset as trainDatasetDenoising
from pathlib2 import Path
from image.my_image import myImage
import os


class testDataset(testDatasetMonai, testDatasetDenoising):

    def __init__(self, args):
        super(testDataset, self).__init__(args)
        self.image = myImage()

        # Convert outputPathTest to Path object if it's a string
        if hasattr(self.args, 'outputPathTest') and isinstance(self.args.outputPathTest, str):
            from pathlib2 import Path
            self.args.outputPathTest = Path(self.args.outputPathTest)

    def write_result(self, outputBatches=None, resultDir=None,
                     description='',
                     sourcePath=None):
        """

        :param resultDir:
        :param description:
        :param sourcePath:
        :return:
        """
        if self.args.quantum_loss:
            input_keys = [key for key in self.data[0] if key.startswith('input')]
        else:
            input_keys = ['input']
        for index, entry in enumerate(input_keys):
            data_i = self.data[self.dataIdx[-1]][entry]

            Case = Path(os.path.dirname(data_i)).name
            # resultDir = os.path.join(os.path.dirname(self.args.outputPathTest.joinpath(Case)),
            #                          os.path.basename(os.path.dirname(data_i)))
            resultDir = os.path.join(os.path.dirname(self.args.outputPathTest.joinpath(Case)), Case)
            #sourcePath = data_i
            sourcePath = os.path.dirname(data_i)
            description = ''
            print(f'Source Path of denoising is \n{sourcePath}')
            # self.processed = data
            self.image.write_image_(im=outputBatches.output[:, index, ...],
                                    path=resultDir,
                                    fixZPositions=self.args.fixZPositions,
                                    description=description,
                                    source_path=sourcePath)

    def write_results_batch(self, outputBatches):
        input_keys = [key for key in self.data[0] if key.startswith('input')]
        for index, entry in enumerate(input_keys):
            data_i = self.data[self.dataIdx[-1]][entry]
            Case = Path(data_i).name
            resultDir = os.path.join(os.path.dirname(self.args.outputPathTest.joinpath(Case)), os.path.basename(os.path.dirname(data_i)), Case)

            sourcePath = data_i
            description = ''
            print(f'Source Path of denoising is \n{sourcePath}')
            #self.processed = data
            indecies= [0, 4, 1, 3, 2]
            img = outputBatches.output.permute(indecies)[0, :, index, ...]
            self.image.write_image_(im=img,
                                    path=resultDir,
                                    fixZPositions=self.args.fixZPositions,
                                    description=description,
                                    source_path=sourcePath)

        #testDatasetDenoising.write_result(self, resultDir=resultDir.as_posix(),
        #                                 sourcePath=sourcePath, mod=index)
        # data_i = self.data[self.dataIdx[-1]]['input']
        # Case = Path(data_i).name
        # resultDir = self.args.outputPathTest.joinpath(Case)
        # sourcePath = data_i
        # print(f'Source Path of denoising is \n{sourcePath}')
        # testDatasetDenoising.write_result(self, resultDir=resultDir.as_posix(),
        #                                   sourcePath=sourcePath)
        # del self.processed


class trainDataset(trainDatasetMonai, trainDatasetDenoising):

    def __init__(self, args):
        super(trainDataset, self).__init__(args)


class testDatasetCached(testDatasetMonaiCached, testDatasetDenoising):

    def __init__(self, args):
        super(testDatasetCached, self).__init__(args)


class trainDatasetCached(trainDatasetMonaiCached, trainDatasetDenoising):

    def __init__(self, args):
        super(trainDatasetCached, self).__init__(args)
