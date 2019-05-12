from sklearn.pipeline import Pipeline


class MLPipeline(Pipeline):

    def __init__(self, name="MLPipeline", steps=[]):

        super(MLPipeline, self).__init__(steps)
        self.name = name
