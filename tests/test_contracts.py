import pytest
from pydantic import ValidationError

from regression_detection.contracts import EmailClassification, SupportEmail


def test_contracts_accept_valid_values():
    assert SupportEmail(text="Need help").text == "Need help"
    result = EmailClassification(category="account", summary="Needs help signing in.")
    assert result.category == "account"


def test_contract_rejects_unknown_category():
    with pytest.raises(ValidationError):
        EmailClassification(category="shipping", summary="Not supported.")
