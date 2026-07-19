"""
OGDD Result Tests

Tests for standard analysis outputs.
"""

from ogdd.core.result import (
    AnalysisResult
)



def test_create_basic_result():

    result = AnalysisResult(
        value="test",
        algorithm="DemoAlgorithm"
    )


    assert (
        result.value
        ==
        "test"
    )


    assert (
        result.algorithm
        ==
        "DemoAlgorithm"
    )



def test_default_version():

    result = AnalysisResult(
        value=None,
        algorithm="Test"
    )


    assert (
        result.version
        ==
        "0.1"
    )



def test_custom_version():

    result = AnalysisResult(
        value=None,
        algorithm="Test",
        version="0.2"
    )


    assert (
        result.version
        ==
        "0.2"
    )



def test_add_metadata():

    result = AnalysisResult(
        value=None,
        algorithm="Test"
    )


    result.add_metadata(
        "points",
        1000
    )


    assert (
        result.metadata["points"]
        ==
        1000
    )



def test_confidence_storage():

    confidence = {
        "score":0.98
    }


    result = AnalysisResult(
        value=None,
        algorithm="Test",
        confidence=confidence
    )


    assert (
        result.confidence["score"]
        ==
        0.98
    )
