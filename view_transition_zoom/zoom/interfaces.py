"""Reserved research extension interfaces; no learned models are implemented."""

from abc import ABC, abstractmethod


class LearnedViewTransition(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class LearnedBlendMask(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class FrameInterpolator(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class NovelViewRenderer(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


__all__ = ["FrameInterpolator", "LearnedBlendMask", "LearnedViewTransition", "NovelViewRenderer"]

