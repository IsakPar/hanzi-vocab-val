"""
Pytest fixtures for vocab-validator tests
"""

import pytest
import json
import os
import tempfile
import shutil
from app.validator import VocabValidator


@pytest.fixture
def mock_curriculum():
    """
    Create mock curriculum data for testing.
    Simulates real curriculum structure with words at different positions.
    """
    return {
        "words": {
            # HSK 1, Lesson 1
            "你好": "hsk1-l1",
            "谢谢": "hsk1-l1",
            "再见": "hsk1-l1",
            # HSK 1, Lesson 2
            "吃": "hsk1-l2",
            "喝": "hsk1-l2",
            "水": "hsk1-l2",
            # HSK 1, Lesson 3
            "学习": "hsk1-l3",
            "工作": "hsk1-l3",
            # HSK 2, Lesson 1
            "可能": "hsk2-l1",
            "应该": "hsk2-l1",
            "需要": "hsk2-l1",
            # HSK 2, Lesson 2
            "虽然": "hsk2-l2",
            "但是": "hsk2-l2",
            # HSK 3, Lesson 1
            "聪明": "hsk3-l1",
            "努力": "hsk3-l1",
        },
        "version": "test-v1"
    }


@pytest.fixture
def temp_data_dir(mock_curriculum):
    """Create a temporary directory with mock curriculum data"""
    temp_dir = tempfile.mkdtemp()
    
    # Write curriculum.json
    curriculum_path = os.path.join(temp_dir, "curriculum.json")
    with open(curriculum_path, "w", encoding="utf-8") as f:
        json.dump(mock_curriculum, f, ensure_ascii=False)
    
    # Write version.txt
    version_path = os.path.join(temp_dir, "version.txt")
    with open(version_path, "w") as f:
        f.write(mock_curriculum["version"])
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def validator(temp_data_dir):
    """Create a validator with test data loaded"""
    v = VocabValidator(data_dir=temp_data_dir)
    v.load()
    return v

