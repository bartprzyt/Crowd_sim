"""Rejestr modeli. Nowy model = nowy plik + dekorator @register("nazwa")."""
MODEL_REGISTRY = {}


def register(name):
    def deco(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return deco


def create_model(name, **params):
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Nieznany model '{name}'. Dostępne: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**params)


from .base import Model
from . import simple
