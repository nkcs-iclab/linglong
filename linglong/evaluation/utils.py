import linglong.evaluation


def get_metric(name: str | None):
    if name is None:
        return None
    return {
        'math23k_dataset_metric': linglong.evaluation.metrics.Math23kDatasetMetric,
    }[name]
