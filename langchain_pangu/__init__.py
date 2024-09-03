from importlib import metadata

from langchain_pangu.chat_models import ChatPanGu
from langchain_pangu.embeddings import PanGuEmbeddings
from langchain_pangu.llms import PanGuLLM

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "ChatPanGu",
    "PanGuEmbeddings",
    "PanGuLLM",
    "__version__",
]
