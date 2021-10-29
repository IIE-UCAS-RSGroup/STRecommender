import importlib

def normalize(scores):
    max_score = max(scores)
    if not max_score == 0:
        scores = [s / max_score for s in scores]
    return scores

def get_model(model_name):
    model_submodule = ['context', 'general', 'spatialtemporal']
    model_module = None
    for submodule in model_submodule:
        module_path = '.'.join(['model', submodule, model_name])
        if importlib.util.find_spec(module_path, __name__):
            model_module = importlib.import_module(module_path, __name__)
            break

    if model_module is None:
        raise ValueError('`model_name` [{}] is not the name of an existing model.'.format(model_name))
    model_class = getattr(model_module, model_name)
    return model_class
