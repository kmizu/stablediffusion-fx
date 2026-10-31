"""Read editor vectors and guidance from the engine's active text encoders."""

from .embedding_editor_state import guidance_markers


def read_editor_vectors(engine, name, references, *, guarded=True):
    """Read active rows/ranges under the engine's offload materialization guard."""
    def read():
        import torch
        from .embedding_inspector import _file_embeddings, _resolve, _streams
        streams = _streams(engine.pipe)
        matches = [token for token in getattr(engine, 'embedding_token_owners', {})
                   if token.casefold() == name.strip().casefold()]
        if len(matches) != 1:
            raise ValueError('loaded embedding ownership is ambiguous or unknown')
        actual = matches[0]
        ownership = engine.embedding_token_owners[actual]
        if set(ownership) != set(streams):
            raise ValueError('loaded embedding ownership is incomplete')
        path = getattr(engine, 'loaded_embedding_files', {}).get(actual)
        if not path:
            raise ValueError('loaded embedding file is unknown')
        load_file, _, _ = _file_embeddings([path], streams)
        file_values = load_file(actual)
        if file_values is None:
            raise ValueError('loaded embedding file is unavailable')
        vectors, ranges = {}, {}
        for key, (tokenizer, encoder) in streams.items():
            weight = encoder.get_input_embeddings().weight
            _, selected = _resolve(tokenizer, weight, actual, {actual: ownership[key]})
            selected_cpu = selected.detach().to(device='cpu', dtype=torch.float32)
            disk = file_values[key].detach().to(device='cpu', dtype=getattr(weight, 'dtype', torch.float32))
            disk = disk.to(device='cpu', dtype=torch.float32)
            if not torch.equal(selected_cpu, disk):
                raise ValueError('embedding file differs from loaded model; reload before editing')
            vectors[key] = selected_cpu.tolist()
            count = min(int(tokenizer.vocab_size), int(weight.shape[0]))
            if count <= 0:
                raise ValueError('tokenizer vocabulary is empty')
            base = weight[:count].detach().to(device='cpu', dtype=torch.float32)
            lows = base.amin(dim=0).tolist()
            highs = base.amax(dim=0).tolist()
            ranges[key] = [[float(low), float(high)] for low, high in zip(lows, highs)]
        guidance = []
        for text, color in references:
            reference = {}
            for key, (tokenizer, encoder) in streams.items():
                ids = tokenizer.encode(text.lower(), add_special_tokens=False)
                if not ids:
                    raise ValueError(f'guidance text has no tokens: {text}')
                _, row = _resolve(tokenizer, encoder.get_input_embeddings().weight, f'#{ids[0]}')
                reference[key] = row.detach().to(device='cpu', dtype=torch.float32).tolist()[0]
            guidance.append({'text': text, 'color': color, 'markers': guidance_markers(ranges, reference)})
        return {'vectors': vectors, 'ranges': ranges, 'guidance': guidance}
    return engine._with_embedding_encoder_access(read) if guarded else read()
