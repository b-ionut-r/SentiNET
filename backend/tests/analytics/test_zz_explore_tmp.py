import pytest
from app.analytics.build import build_analysis
from tests.analytics.render import render
from tests.analytics.scenarios import ALL

@pytest.mark.parametrize("name", list(ALL))
def test_explore(name):
    print("\n" + "=" * 100 + "\n" + render(build_analysis(ALL[name]())))
