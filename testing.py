from refactor import parse_product_data
import pytest
import json

def test_parse_product_data_extracts_category():
    # 1. Arrange: Load the actual data from your file
    with open("raw_product_mk1.json", "r") as f:
        data = json.load(f)
    
    # REACH INSIDE: Get the first product dictionary from the "Products" list
    
    # 2. Act: Pass that dictionary into your function
    result = parse_product_data(data)
    
    # 3. Assert: Verify the results
    assert result["product_id"] == "9N7271QN4SGB"
    assert result["category"] == "Fighting"
    assert "BloGor" in result["esrb_descriptors"]
    assert "StrLan" in result["esrb_descriptors"]

    


def test_parse_product_data_extracts_esrb():
    # Arrange: Mocking specifically for the ESRB logic we discussed
    mock_product = {
        "MarketProperties": [{
            "ContentRatings": [
                {
                    "RatingSystem": "ESRB",
                    "RatingDescriptors": ["ESRB:Blood", "ESRB:Violence"]
                }
            ]
        }]
    }
    
    # Act
    result = parse_product_data(mock_product)
    
    # Assert
    assert "Blood" in result["esrb_descriptors"]
    assert "ESRB:" not in result["esrb_descriptors"][0] # Verify cleaning logic