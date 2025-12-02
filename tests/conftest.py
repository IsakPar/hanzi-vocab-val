"""
Pytest fixtures for vocab-validator tests
"""

import pytest
import json
import os
import tempfile
import shutil
from fastapi.testclient import TestClient
from app.validator import VocabValidator


# ═══════════════════════════════════════════════════════════
# Shared API Client Fixtures
# ═══════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def mock_data_dir():
    """Create mock data directory before app import"""
    temp_dir = tempfile.mkdtemp()
    
    # Create comprehensive curriculum for realistic tests
    curriculum = {
        "words": {
            # HSK 1, Lesson 1-5
            "你": "hsk1-l1",
            "好": "hsk1-l1",
            "你好": "hsk1-l1",
            "我": "hsk1-l1",
            "是": "hsk1-l1",
            "谢谢": "hsk1-l1",
            "再见": "hsk1-l1",
            "他": "hsk1-l1",
            "她": "hsk1-l1",
            "吃": "hsk1-l2",
            "喝": "hsk1-l2",
            "水": "hsk1-l2",
            "饭": "hsk1-l2",
            "看": "hsk1-l2",
            "学习": "hsk1-l3",
            "工作": "hsk1-l3",
            "学生": "hsk1-l3",
            "老师": "hsk1-l3",
            "中文": "hsk1-l4",
            "中国": "hsk1-l4",
            "人": "hsk1-l4",
            "说": "hsk1-l5",
            "汉语": "hsk1-l5",
            "朋友": "hsk1-l6",
            "今天": "hsk1-l7",
            "明天": "hsk1-l8",
            "去": "hsk1-l9",
            "来": "hsk1-l10",
            # HSK 2
            "可能": "hsk2-l1",
            "图书馆": "hsk2-l1",
            "应该": "hsk2-l1",
            "需要": "hsk2-l1",
            "能": "hsk2-l1",
            "考试": "hsk2-l2",
            "虽然": "hsk2-l2",
            "但是": "hsk2-l2",
        },
        "version": "test-v1"
    }
    
    # Write curriculum.json
    curriculum_path = os.path.join(temp_dir, "curriculum.json")
    with open(curriculum_path, "w", encoding="utf-8") as f:
        json.dump(curriculum, f, ensure_ascii=False)
    
    # Write version.txt
    version_path = os.path.join(temp_dir, "version.txt")
    with open(version_path, "w") as f:
        f.write("test-v1")
    
    yield temp_dir
    
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="module")
def client(mock_data_dir):
    """Create test client with mock data"""
    import jieba
    
    from app.main import app, validator
    
    # Load curriculum into validator
    validator.data_dir = mock_data_dir
    validator.curriculum = {}
    validator.loaded = False
    validator.load()
    
    # Add words to jieba for proper segmentation
    for word in validator.curriculum.keys():
        jieba.add_word(word, freq=1000)
    
    return TestClient(app)


# ═══════════════════════════════════════════════════════════
# Unit Test Fixtures  
# ═══════════════════════════════════════════════════════════

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

