from stability_adalora.compat import EXPECTED_PEFT_VERSION, assert_peft_compat, installed_peft_version


def test_expected_peft_version_is_pinned():
    assert EXPECTED_PEFT_VERSION == "0.20.0"


def test_installed_peft_matches_project_target():
    assert installed_peft_version() == EXPECTED_PEFT_VERSION
    assert assert_peft_compat() == EXPECTED_PEFT_VERSION
