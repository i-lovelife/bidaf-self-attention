# pylint: disable=no-self-use,invalid-name
import pytest

from allennlp.common import Params
from allennlp.common.util import ensure_list

from src.common.file_root import FIXTURES_ROOT
from src.dataset_readers.squad_v2 import SquadReaderV2

class TestSquadReaderV2:
    @pytest.mark.parametrize("lazy", (True, False))
    def test_read_from_file(self, lazy):
        reader = SquadReaderV2(lazy=lazy)
        instances = ensure_list(reader.read(FIXTURES_ROOT / 'squad-v2.0.json'))
        assert len(instances) == 4

        assert [t.text for t in instances[0].fields["question"].tokens[:3]] == ["When", "were", "the"]
        assert [t.text for t in instances[0].fields["passage"].tokens[:3]] == ["The", "Normans", "("]
        assert [t.text for t in instances[0].fields["passage"].tokens[-3:]] == ["succeeding", "centuries", "."]

        assert [t.text for t in instances[2].fields["question"].tokens[:3]] == ["What", "is", "France"]
        assert [t.text for t in instances[2].fields["passage"].tokens[:3]] == ["The", "Normans", "("]
        assert [t.text for t in instances[2].fields["passage"].tokens[-3:]] == ["succeeding", "centuries", "."]
        # Todo: add test for plausible_answers

    def test_can_build_from_params(self):
        reader = SquadReaderV2.from_params(Params({}))
        # pylint: disable=protected-access
        assert reader._tokenizer.__class__.__name__ == 'WordTokenizer'
        assert reader._token_indexers["tokens"].__class__.__name__ == 'SingleIdTokenIndexer'
