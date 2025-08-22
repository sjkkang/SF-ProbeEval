"""
SF-ProbeEval: Science Fiction Text Probing Evaluation Dataset and Pipeline

A comprehensive toolkit for evaluating language model understanding of science fiction narratives
through topic modeling, text infilling, and expert annotation.
"""

__version__ = "1.0.0"
__author__ = "Sujin Kang"
__license__ = "MIT"

from . import sf_topic_pipeline, evaluation

__all__ = ["sf_topic_pipeline", "evaluation"]
