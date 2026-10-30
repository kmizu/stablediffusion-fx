import numpy as np
import pytest
from sdfx.embedding_merge_export import export_embeddings
from sdfx.embedding_merge_files import select_path, requested_paths


def test_source_conversion_siblings_resolve_by_model_width(tmp_path):
    paths = export_embeddings(tmp_path, 'mixed', {'clip_l': np.ones((2, 768))})
    assert select_path('mixed', paths, {'clip_l': 768}) == str(paths[0])
    assert select_path('mixed', paths, {'clip_l': 768, 'clip_g': 1280}) == str(paths[1])
    assert requested_paths(paths, ["<'mixed'>"], {'clip_l': 768}) == [str(paths[0])]


def test_two_compatible_files_stay_ambiguous_and_unreferenced_names_are_untouched(tmp_path):
    a = export_embeddings(tmp_path / 'a', 'mixed', {'clip_l': np.ones((1, 768))})[0]
    b = export_embeddings(tmp_path / 'b', 'mixed', {'clip_l': np.ones((1, 768))})[0]
    with pytest.raises(ValueError, match='ambiguous'):
        select_path('mixed', [a, b], {'clip_l': 768})
    assert requested_paths([a, b], ['unrelated'], {'clip_l': 768}) == [a, b]


def test_sd2_width_is_not_mistaken_for_sd1_or_sdxl_and_pt_probe_is_explicit():
    probe = lambda path: {'emb_params': [2, 1024] if str(path).startswith('sd2') else [2, 768]}
    assert select_path('style', ['sd1/style.pt', 'sd2/style.pt'], {'clip_l': 1024}, probe=probe) == 'sd2\\style.pt'
    with pytest.raises(ValueError, match='compatible'):
        select_path('style', ['sd2/style.pt'], {'clip_l': 768}, probe=probe)


def test_requested_catalog_does_not_infer_names_from_substring(tmp_path):
    paths = export_embeddings(tmp_path, 'cat', {'clip_l': np.ones((1, 768))})
    assert requested_paths(paths, ['cathedral'], {'clip_l': 768}) == paths
